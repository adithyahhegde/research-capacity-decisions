import os, json
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

OUT = Path('research_capacity/results')
OUT.mkdir(parents=True, exist_ok=True)
YEAR = 2024
urls = [f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{YEAR}-{m:02d}.parquet' for m in range(1, 13)]

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs")

parts = []
for i, u in enumerate(urls, 1):
    print(f'[{i}/12] {u}', flush=True)
    q = """
        SELECT date_trunc('hour', tpep_pickup_datetime) AS hour,
               PULocationID, count(*)::INTEGER AS demand
        FROM read_parquet(?)
        WHERE PULocationID IS NOT NULL
          AND tpep_pickup_datetime >= TIMESTAMP '2024-01-01 00:00:00'
          AND tpep_pickup_datetime < TIMESTAMP '2025-01-01 00:00:00'
        GROUP BY 1, 2
    """
    parts.append(con.execute(q, [u]).df())

hourly = (pd.concat(parts, ignore_index=True)
            .groupby(['hour', 'PULocationID'], as_index=False)['demand'].sum())
hourly['hour'] = pd.to_datetime(hourly['hour'])
hourly = hourly.sort_values(['hour', 'PULocationID']).reset_index(drop=True)

assert hourly.hour.min() >= pd.Timestamp('2024-01-01')
assert hourly.hour.max() < pd.Timestamp('2025-01-01')
assert hourly.demand.sum() > 0
hourly.to_csv(OUT / 'hourly_demand_2024.csv', index=False)

start = pd.Timestamp('2024-01-01 00:00:00')
end = pd.Timestamp('2024-12-31 23:00:00')
top = (hourly[hourly.hour < start + (end - start) * 0.60]
       .groupby('PULocationID').demand.sum().nlargest(20).index.tolist())

models = {
    'Ridge': Ridge(alpha=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2),
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=.05, subsample=.9,
                            colsample_bytree=.9, objective='reg:squarederror', random_state=42, n_jobs=-1)
}

def smape(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    denom = np.abs(y) + np.abs(p); numer = 2.0 * np.abs(y - p)
    out = np.zeros_like(numer, dtype=float)
    np.divide(numer, denom, out=out, where=denom != 0)
    return float(np.mean(out) * 100)

val, test, preds = [], [], []
features = ['hour_of_day', 'dow', 'month', 'weekend', 'lag1', 'lag2', 'lag24', 'lag48', 'lag168', 'roll24']

for zone in top:
    raw = hourly[hourly.PULocationID == zone][['hour', 'demand']].copy()
    idx = pd.date_range(start, end, freq='h')
    z = raw.set_index('hour').reindex(idx).rename_axis('hour').reset_index()
    z['demand'] = z['demand'].fillna(0.0)
    z['hour_of_day'] = z.hour.dt.hour; z['dow'] = z.hour.dt.dayofweek
    z['month'] = z.hour.dt.month; z['weekend'] = (z.dow >= 5).astype(int)
    for lag in [1, 2, 24, 48, 168]: z[f'lag{lag}'] = z.demand.shift(lag)
    z['roll24'] = z.demand.shift(1).rolling(24).mean(); z = z.dropna().reset_index(drop=True)
    n = len(z); a = int(.60*n); b = int(.80*n); tr, va, te = z.iloc[:a], z.iloc[a:b], z.iloc[b:]

    pv = va['lag168'].to_numpy()
    val.append({'zone': zone, 'model': 'Seasonal Naive', 'MAE': mean_absolute_error(va.demand,pv),
                'RMSE': mean_squared_error(va.demand,pv)**.5, 'sMAPE': smape(va.demand.to_numpy(),pv)})
    for name, template in models.items():
        m = template.__class__(**template.get_params()); m.fit(tr[features],tr.demand); p = m.predict(va[features])
        val.append({'zone':zone,'model':name,'MAE':mean_absolute_error(va.demand,p),
                    'RMSE':mean_squared_error(va.demand,p)**.5,'sMAPE':smape(va.demand.to_numpy(),p)})

    tv = pd.concat([tr,va],ignore_index=True)
    for name, template in models.items():
        m = template.__class__(**template.get_params()); m.fit(tv[features],tv.demand); p=m.predict(te[features])
        test.append({'zone':zone,'model':name,'MAE':mean_absolute_error(te.demand,p),
                     'RMSE':mean_squared_error(te.demand,p)**.5,'sMAPE':smape(te.demand.to_numpy(),p)})
        preds.append(pd.DataFrame({'hour':te.hour.to_numpy(),'zone':zone,'actual':te.demand.to_numpy(),'model':name,'forecast':p}))
    preds.append(pd.DataFrame({'hour':te.hour.to_numpy(),'zone':zone,'actual':te.demand.to_numpy(),
                               'model':'Seasonal Naive','forecast':te['lag168'].to_numpy()}))

val = pd.DataFrame(val); test = pd.DataFrame(test); pred = pd.concat(preds,ignore_index=True)
assert val[['MAE','RMSE']].to_numpy().mean() > .01 and test[['MAE','RMSE']].to_numpy().mean() > .01
assert pred.actual.max() > 0
val.to_csv(OUT/'validation_accuracy_2024.csv',index=False); test.to_csv(OUT/'test_forecast_accuracy_2024.csv',index=False)
pred.to_csv(OUT/'test_predictions_2024.csv',index=False)

rows=[]
for model,g in pred.groupby('model'):
    actual=g.actual.to_numpy(); forecast=g.forecast.to_numpy()
    for beta in [0,.1,.2,.3]:
        cap=np.ceil(np.maximum(forecast,0)*(1+beta)); under=np.maximum(actual-cap,0); over=np.maximum(cap-actual,0)
        for r in [1,3,5,10]:
            rows.append({'model':model,'buffer':beta,'under_over_ratio':r,'mean_cost':float(np.mean(r*under+over)),
                         'under_rate':float(np.mean(under>0)),'mean_under':float(under.mean()),'mean_over':float(over.mean())})
ops=pd.DataFrame(rows); ops.to_csv(OUT/'operational_sensitivity_2024.csv',index=False)
best=[]
for (model,ratio),g in ops.groupby(['model','under_over_ratio']): best.append(g.loc[g.mean_cost.idxmin()].to_dict())
summary={'months':12,'zones':20,'date_min':str(hourly.hour.min()),'date_max':str(hourly.hour.max()),
         'total_trips_aggregated':int(hourly.demand.sum()),
         'validation_mean':val.groupby('model')[['MAE','RMSE','sMAPE']].mean().round(4).to_dict('index'),
         'test_mean':test.groupby('model')[['MAE','RMSE','sMAPE']].mean().round(4).to_dict('index'),
         'best_operational':best}
(OUT/'full_year_summary.json').write_text(json.dumps(summary,indent=2,default=float)); print(json.dumps(summary,indent=2,default=float))
