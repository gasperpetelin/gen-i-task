# GEN-I Recruitment Tasks

This repository contains solutions to three tasks covering hydro production forecasting, electricity price analysis, and price scenario modelling. Data manipulation is done with [Polars](https://pola.rs), classical ML models are built with [scikit-learn](https://scikit-learn.org), and automated model selection with [AutoGluon](https://auto.gluon.ai).

---

## Task 1: Hydro Production Forecast Model

### 1.a: Linear model

We fit a linear relationship between river inflow and daily energy production:

$$y = f(x) = k \cdot x + n$$

where $x$ is river inflow [m³/s] and $y$ is realized daily production [MWh/day].

Coefficients are estimated with **Ordinary Least Squares (OLS)**, unregularised, minimising the sum of squared residuals.

**Results:**

| Parameter | Value |
|---|---|
| k (slope) | 2.4841 MWh/day per m³/s |
| n (intercept) | 1935.09 MWh/day |
| R² (all data) | 0.9135 |

**Predictions:**

| River inflow | Predicted production |
|---|---|
| 4 000 m³/s | 11 871 MWh/day |
| 11 000 m³/s | 29 260 MWh/day |

![Linear model](figures/task_1a_linear_model.png)

---

### 1.b: Non-linear models

We compare three models on a **chronological 80/20 train-test split**:

| Model | Description |
|---|---|
| **Linear** | Baseline OLS straight-line fit |
| **Spline** | Piecewise cubic spline basis (8 knots) + linear regression, captures smooth non-linearity |
| **AutoGluon** | Automated ML, trains and ensembles many model types, selects the best combination |

**Test set metrics:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear | 1011.6 | 781.3 | 0.9073 |
| Spline | 938.2 | 725.3 | 0.9202 |
| AutoGluon | 887.0 | 693.3 | 0.9287 |

**AutoGluon achieves the best performance on the held-out test set**, with the spline capturing the saturation effect at high flows that the linear model misses.

![Test set: actual vs predicted](figures/task_1b_timeseries.png)

![Model curves](figures/task_1b_model_curves.png)

---

### 1.c: Daily forecasting process

The model above uses measured river flow as input, in a real forecasting scenario this isn't known yet, so it needs to be forecast as well. The key inputs would be:

- **Forecasted river flow**: the main driver. Comes from hydrological models fed by weather forecasts (precipitation, temperature, snowmelt).
- **Weather forecasts**: precipitation and temperature drive river flow upstream; temperature also affects demand and plant efficiency.
- **Current river state**: today's measured flow and upstream gauge readings to ground the forecast in the latest observations.
- **Planned outages or constraints**: maintenance windows or grid operator requests that limit production regardless of available flow.
- **Lag features**: since historical values are always known at forecast time, they can be added directly as inputs:
  - Recent production (yesterday, last 3 and 7 days), the plant doesn't jump from one operating regime to another overnight
  - Recent river flow averages (last 3 and 7 days), captures whether the basin is in a wet or drought period, since water already upstream takes time to arrive at the generators

Each day we would pull the latest forecasts, compute the lags from yesterday's actuals, and run the model to get a production estimate for the next few days.

---

### 1.d: Factors influencing electricity prices

**Daily price drivers**

Apart from hydro, the main daily drivers are:

- **Demand**: the most predictable component. Load follows a daily and weekly pattern (morning/evening peaks, low nights and weekends) and is strongly driven by temperature through heating and cooling demand.
- **Gas and coal prices**: gas plants are typically the last ones switched on to meet demand, so their cost largely sets the market price. CO₂ allowance costs add on top.
- **Wind and solar output**: wind and solar produce power essentially for free once built, so when a lot of it is available it pushes gas and coal plants out and pulls prices down. Forecast errors in renewables are the main source of intraday volatility.
- **Cross-border flows**: when transmission lines are full, electricity can't flow freely between countries, so a local shortage or surplus can cause big price swings that neighbouring markets can't help with.
- **Unplanned outages**: if a large power plant goes down unexpectedly, supply drops fast and prices spike.

**Expected changes**

The direction is fairly clear: more renewables mean lower average prices but wilder swings. Some things I expect in the coming years:

- More hours of zero or negative prices, especially around midday when solar peaks (just like in Germany).
- Sharper price spikes during low-wind, low-solar periods.
- The gap between cheap and expensive electricity is growing, midday power (when solar is abundant) is getting cheaper, while morning and evening power (when solar disappears and everyone uses electricity at once) is getting more expensive
- Batteries and flexible demand (EVs, heat pumps) starting to smooth things out, but probably not fast enough to keep up with the pace of renewable growth in the near term
- Short-term forecasting becoming more valuable for both plant operators and traders, getting the next few hours or days right matters more when prices are volatile

---

## Task 2: Price Value of Electricity

### 2.a: Baseload price, full year and monthly

The **baseload price** is the simple arithmetic mean of all hourly prices in the year calculated as:

```python
# Annual
baseload_annual = df["price"].mean()  # 48.75 €/MWh

# Monthly
monthly_baseload = (
    df
    .with_columns(pl.col("datetime_cet").dt.month().alias("month"))
    .group_by("month")
    .agg(pl.col("price").mean().alias("baseload_€MWh"))
    .sort("month")
)
```

**Annual baseload value 2019: 48.75 €/MWh**

Monthly values are higher in winter and lower in spring/autumn:

![Monthly baseload price 2019](figures/task_2a_baseload_monthly.png)

---

### 2.b: Peakload price, monthly

The **peakload price** is the volume-weighted average over peak hours only: **08:00-20:00, Monday-Friday** (12 hours/day * 5 days = 60 hours/week).

```python
df_peak = df.with_columns([
    pl.col("datetime_cet").dt.month().alias("month"),
    pl.col("datetime_cet").dt.hour().alias("hour"),
    pl.col("datetime_cet").dt.weekday().alias("weekday"),  # 1=Mon … 7=Sun
])

peak_mask = (
    (pl.col("hour") >= 8) & (pl.col("hour") <= 19) &
    (pl.col("weekday") <= 5)  # Mon-Fri
)

monthly_peakload = (
    df_peak
    .filter(peak_mask)
    .group_by("month")
    .agg(pl.col("price").mean().alias("peakload_€MWh"))
    .sort("month")
)
```

Peakload (orange) is consistently above baseload (blue) each month:

![Monthly baseload vs peakload price 2019](figures/task_2b_peakload_monthly.png)

---

### 2.c: Total production per source and Consumer X consumption

Production columns (`solar`, `hydro`, `wind`, `nuclear`, `lignite`) are already in MWh. Consumer X is recorded in kWh and is converted to MWh for comparison.

**Annual totals 2019:**

| Source | Total [MWh] |
|---|---|
| Solar | 245,268 |
| Hydro | 4,309,297 |
| Wind | 4,636 |
| Nuclear | 5,499,420 |
| Lignite | 3,997,051 |
| Consumer X | 8,843 |

Nuclear is the dominant source, followed by hydro and lignite. Wind and solar contribute far less in this market. Consumer X draws 8.8 GWh over the year, roughly 0.06 % of total generation.

---

### 2.d: Average price value per source and Consumer X

The **value** is the average price each source actually captured, weighted by hourly output.

All sources sit close to the baseload reference of 48.75 €/MWh. Lignite captures the highest value because it dispatches during expensive hours. Nuclear captures the least, it runs flat around the clock including cheap overnight hours, which pulls its average down. Solar and Consumer X both land above baseload, meaning solar generates (and Consumer X consumes) relatively more during higher-priced hours.

![Volume-weighted average captured price by source 2019](figures/task_2d_value_by_source.png)

---

## Task 3: Price Scenarios

### 3.a: Expected value on 1.4.2020

100 daily price scenarios from 17.1.2020 to 30.4.2020. Expected value = mean across all 100 scenarios for that day.

**Expected price on 1.4.2020: 48.14 €/MWh**

![All 100 price scenarios and histogram on 1.4.2020](figures/task_3a_scenarios.png)

---

### 3.b: Algorithm profit in scenario #77

The algorithm has perfect next-day foresight and can hold at most 1 unit. The greedy strategy is: **buy** whenever tomorrow's price is higher than today's, **sell** whenever it is lower or equal. Any open position is closed on the last day.

**Scenario #77 total profit: 27.15 €/MWh**

The top panel shows the price path with buy and sell markers. The bottom panel tracks the cumulative P&L, it rises steadily and never dips below zero since the algorithm only enters positions it knows will be profitable.

![Scenario #77 trades and cumulative P&L](figures/task_3b_scenario77.png)

---

### 3.c: Profit across all 100 scenarios

Same algorithm run on all 100 scenarios:

| | Profit [€/MWh] |
|---|---|
| Best scenario (Scenarij_48) | 39.87 |
| Mean across all scenarios | 24.09 |
| Worst scenario | 14.06 |

Left: total profit per scenario (each bar = one scenario). Right: price path of the best scenario (Scenarij_48).

![Algorithm profit per scenario and best-case path](figures/task_3c_profits.png)

---

### 3.d: Value of the call option

A **European call option** gives the holder the right, but not the obligation, to buy electricity on 1.4.2020 at a fixed **strike price of 55 €/MWh**. The payoff at expiry is:

$$\text{payoff} = \max(0,\ P_{\text{1.4.2020}} - 55)$$

If the market price exceeds the strike the option is exercised and the holder profits by the difference. If the market price is below the strike the option expires worthless and the holder loses only the premium paid upfront.

The **fair value** of the option is the mean payoff across all 100 scenarios:

```python
strike = 55.0
prices_april1 = scenario_prices.filter(date == "2020-04-01")
option_value = np.maximum(0, prices_april1 - strike).mean()  # 0.31 €/MWh
```

**Results:**

| Metric | Value |
|---|---|
| Option value | **0.31 €/MWh** |
| Scenarios above strike | **7 / 100** |
| Payoff range | 0.00 - 7.41 €/MWh |

The option is cheap because the expected price on 1.4.2020 is 48 €/MWh, well below the 55 €/MWh strike. Only 7 out of 100 scenarios are above the strike, so most of the time the option expires worthless. Those 7 scenarios produce payoffs up to 7.41 €/MWh, on average 0.31 €/MWh.

The left panel shows the payoff distribution: the grey bar represents the 93 payoff == 0 and the orange bars show the small number of payoff>0 scenarios. The right panel overlays the strike and expected price on the full price distribution for 1.4.2020, showing how far into the tail the strike sits.

![Call option payoff distribution and price distribution on 1.4.2020](figures/task_3d_option.png)
