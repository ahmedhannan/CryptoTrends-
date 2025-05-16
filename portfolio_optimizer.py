import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, INITIAL_CAPITAL, TRANSACTION_COST_PCT, PREDICTION_HORIZON

class PortfolioOptimizer:
    def __init__(self, list_of_cryptos, initial_portfolio_capital=INITIAL_CAPITAL, cost_per_transaction_pct=TRANSACTION_COST_PCT):
        self.crypto_symbols_list = list_of_cryptos
        self.current_asset_holdings_units = {crypto: 0.0 for crypto in list_of_cryptos}
        self.cash_balance = initial_portfolio_capital
        self.total_portfolio_value_pv = initial_portfolio_capital
        self.transaction_cost_percentage = cost_per_transaction_pct
        self.current_asset_values = {crypto: 0.0 for crypto in list_of_cryptos}
        self.current_percentage_allocations = {crypto: 0.0 for crypto in list_of_cryptos}
        self._update_portfolio_metrics_and_allocations({})

    def _update_portfolio_metrics_and_allocations(self, latest_asset_prices_dict):
        self.total_portfolio_value_pv = self.cash_balance
        for crypto_sym in self.crypto_symbols_list:
            price = latest_asset_prices_dict.get(crypto_sym, 0)
            self.current_asset_values[crypto_sym] = self.current_asset_holdings_units[crypto_sym] * price
            self.total_portfolio_value_pv += self.current_asset_values[crypto_sym]
        if self.total_portfolio_value_pv > 0:
            for crypto_sym in self.crypto_symbols_list:
                self.current_percentage_allocations[crypto_sym] = self.current_asset_values[crypto_sym] / self.total_portfolio_value_pv
        else:
            for crypto_sym in self.crypto_symbols_list:
                self.current_percentage_allocations[crypto_sym] = 0.0
        return self.total_portfolio_value_pv

    def get_current_portfolio_summary(self, latest_asset_prices_dict):
        self._update_portfolio_metrics_and_allocations(latest_asset_prices_dict)
        summary = {
            "total_portfolio_value": self.total_portfolio_value_pv,
            "cash_balance": self.cash_balance,
            "asset_values": self.current_asset_values.copy(),
            "asset_units": self.current_asset_holdings_units.copy(),
            "percentage_allocations": self.current_percentage_allocations.copy()
        }
        return summary

    def _calculate_risk_adjusted_scores(self, crypto_price_predictions_dict, asset_historical_volatility_dict, latest_asset_prices_dict):
        risk_scores = {}
        hours_per_year = 252 * 24  # Annualization based on hours

        for crypto_sym, predicted_prices_df in crypto_price_predictions_dict.items():
            if predicted_prices_df.empty or crypto_sym not in latest_asset_prices_dict or crypto_sym not in asset_historical_volatility_dict:
                risk_scores[crypto_sym] = -float('inf')
                continue

            predicted_price_at_horizon_end = predicted_prices_df.iloc[-1, 0]
            current_price = latest_asset_prices_dict[crypto_sym]
            annualized_vol = asset_historical_volatility_dict[crypto_sym]

            if current_price <= 0 or annualized_vol <= 0.01:  # Prevent division by zero or low volatility
                risk_scores[crypto_sym] = -float('inf')
                continue

            expected_return_over_horizon = (predicted_price_at_horizon_end - current_price) / current_price
            # Cap extreme returns
            expected_return_over_horizon = np.clip(expected_return_over_horizon, -0.1, 0.1)

            # Annualize return (PREDICTION_HORIZON is in hours)
            annualized_expected_return = (1 + expected_return_over_horizon)**(hours_per_year / PREDICTION_HORIZON) - 1

            sharpe_like_ratio = (annualized_expected_return - RISK_FREE_RATE) / annualized_vol
            risk_scores[crypto_sym] = sharpe_like_ratio

            print(f"DEBUG: {crypto_sym}: Return={expected_return_over_horizon*100:.2f}%, Annualized Return={annualized_expected_return*100:.2f}%, Volatility={annualized_vol*100:.2f}%, Score={sharpe_like_ratio:.3f}")

        return risk_scores

    def generate_target_percentage_allocations(self, crypto_price_predictions_dict, asset_historical_volatility_dict,
                                             latest_asset_prices_dict, crypto_anomaly_status_dict):
        risk_adjusted_scores = self._calculate_risk_adjusted_scores(
            crypto_price_predictions_dict, asset_historical_volatility_dict, latest_asset_prices_dict
        )

        target_percentage_allocations = {crypto: 0.0 for crypto in self.crypto_symbols_list}
        sum_of_positive_eligible_scores = 0.0
        eligible_cryptos_for_positive_allocation = {}

        print("\n--- Portfolio Optimization: Generating Target Allocations ---")
        print("Calculated Risk-Adjusted Scores (higher is better):")
        for crypto_sym, score in risk_adjusted_scores.items():
            is_anomaly_present = crypto_anomaly_status_dict.get(crypto_sym, False)
            print(f"  {crypto_sym.ljust(15)}: Score={score:.3f}, Anomaly Present={is_anomaly_present}")
            if score > 0 and not is_anomaly_present:
                eligible_cryptos_for_positive_allocation[crypto_sym] = score
                sum_of_positive_eligible_scores += score
            elif is_anomaly_present:
                print(f"  INFO: {crypto_sym} has an anomaly, will not be considered for positive allocation.")
            elif score <= 0:
                print(f"  INFO: {crypto_sym} has non-positive score, will not be considered for positive allocation.")

        if sum_of_positive_eligible_scores > 0:
            for crypto_sym, score in eligible_cryptos_for_positive_allocation.items():
                target_percentage_allocations[crypto_sym] = score / sum_of_positive_eligible_scores
        else:
            print("INFO: No assets eligible for positive allocation (due to scores or anomalies). Suggest holding cash.")

        print("\nSuggested Target Percentage Allocations:")
        total_allocation_pct = 0
        for crypto_sym, alloc_pct in target_percentage_allocations.items():
            print(f"  {crypto_sym.ljust(15)}: {alloc_pct*100:.2f}%")
            total_allocation_pct += alloc_pct
        print(f"  {'Cash'.ljust(15)}: {(1-total_allocation_pct)*100:.2f}% (implied)")

        return target_percentage_allocations

    def simulate_rebalance_to_targets(self, target_percentage_allocations_dict, latest_asset_prices_dict):
        self._update_portfolio_metrics_and_allocations(latest_asset_prices_dict)
        print(f"\n--- Simulating Rebalance ---")
        print(f"Current Total Portfolio Value: ${self.total_portfolio_value_pv:.2f}")
        print("Current Allocations (before rebalance):")
        for crypto_sym, alloc_pct in self.current_percentage_allocations.items():
            print(f"  {crypto_sym.ljust(15)}: {alloc_pct*100:.2f}% (Value: ${self.current_asset_values[crypto_sym]:.2f})")

        suggested_trades_log = []
        target_asset_dollar_values = {
            crypto: self.total_portfolio_value_pv * target_percentage_allocations_dict.get(crypto, 0)
            for crypto in self.crypto_symbols_list
        }

        for crypto_sym in self.crypto_symbols_list:
            current_price = latest_asset_prices_dict.get(crypto_sym)
            if current_price is None or current_price <= 0:
                print(f"WARNING: Cannot rebalance {crypto_sym}, invalid current price: {current_price}")
                continue
            current_dollar_value = self.current_asset_values[crypto_sym]
            target_dollar_value = target_asset_dollar_values.get(crypto_sym, 0)
            if current_dollar_value > target_dollar_value:
                value_to_sell = current_dollar_value - target_dollar_value
                units_to_sell = value_to_sell / current_price
                self.current_asset_holdings_units[crypto_sym] -= units_to_sell
                proceeds_from_sale = value_to_sell * (1 - self.transaction_cost_percentage)
                self.cash_balance += proceeds_from_sale
                trade_msg = (f"SELL {units_to_sell:.6f} units of {crypto_sym} "
                             f"(approx ${value_to_sell:.2f}, proceeds ${proceeds_from_sale:.2f} after costs)")
                suggested_trades_log.append(trade_msg)
                print(f"  {trade_msg}")
        self._update_portfolio_metrics_and_allocations(latest_asset_prices_dict)
        for crypto_sym in self.crypto_symbols_list:
            current_price = latest_asset_prices_dict.get(crypto_sym)
            if current_price is None or current_price <= 0: continue
            current_dollar_value = self.current_asset_holdings_units[crypto_sym] * current_price
            target_dollar_value = self.total_portfolio_value_pv * target_percentage_allocations_dict.get(crypto_sym, 0)
            if current_dollar_value < target_dollar_value:
                value_to_buy = target_dollar_value - current_dollar_value
                units_to_buy = value_to_buy / current_price
                cost_of_purchase_with_fees = value_to_buy * (1 + self.transaction_cost_percentage)
                if self.cash_balance >= cost_of_purchase_with_fees:
                    self.current_asset_holdings_units[crypto_sym] += units_to_buy
                    self.cash_balance -= cost_of_purchase_with_fees
                    trade_msg = (f"BUY {units_to_buy:.6f} units of {crypto_sym} "
                                 f"(approx ${value_to_buy:.2f}, cost ${cost_of_purchase_with_fees:.2f} with fees)")
                    suggested_trades_log.append(trade_msg)
                    print(f"  {trade_msg}")
                else:
                    partially_affordable_value = self.cash_balance / (1 + self.transaction_cost_percentage)
                    if partially_affordable_value > 1:
                        units_can_buy = partially_affordable_value / current_price
                        self.current_asset_holdings_units[crypto_sym] += units_can_buy
                        self.cash_balance -= partially_affordable_value * (1 + self.transaction_cost_percentage)
                        trade_msg = (f"PARTIAL BUY (low cash) {units_can_buy:.6f} units of {crypto_sym} "
                                     f"(approx ${partially_affordable_value:.2f})")
                        suggested_trades_log.append(trade_msg)
                        print(f"  {trade_msg}")
                    else:
                        print(f"  SKIPPED BUY for {crypto_sym}: Insufficient cash (Need ${cost_of_purchase_with_fees:.2f}, Have ${self.cash_balance:.2f})")
        final_pv = self._update_portfolio_metrics_and_allocations(latest_asset_prices_dict)
        print(f"\nPortfolio Rebalance Simulation Complete.")
        print(f"Final Simulated Portfolio Value: ${final_pv:.2f}")
        print(f"Final Simulated Cash Balance: ${self.cash_balance:.2f}")
        print("Final Simulated Allocations (after rebalance attempt):")
        for crypto_sym, alloc_pct in self.current_percentage_allocations.items():
            print(f"  {crypto_sym.ljust(15)}: {alloc_pct*100:.2f}% (Value: ${self.current_asset_values[crypto_sym]:.2f})")
        return suggested_trades_log

if __name__ == '__main__':
    from config import TARGET_CRYPTOS as cryptos_to_manage_test
    from config import PREDICTION_HORIZON as pred_horizon_test
    if not cryptos_to_manage_test:
        print("Please define TARGET_CRYPTOS in config.py for portfolio optimizer test.")
        exit()
    optimizer_instance = PortfolioOptimizer(cryptos_to_manage_test, initial_portfolio_capital=100000)
    dummy_latest_prices = {}
    dummy_predictions = {}
    dummy_volatilities = {}
    dummy_anomaly_statuses = {}
    if 'Bitcoin' in cryptos_to_manage_test:
        dummy_latest_prices['Bitcoin'] = 40000.00
        btc_pred_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=pred_horizon_test, freq='D')
        dummy_predictions['Bitcoin'] = pd.DataFrame(
            {'predicted_close_price': np.linspace(41000, 43500, pred_horizon_test)},
            index=btc_pred_dates
        )
        dummy_volatilities['Bitcoin'] = 0.60
        dummy_anomaly_statuses['Bitcoin'] = False
    if 'Ethereum' in cryptos_to_manage_test:
        dummy_latest_prices['Ethereum'] = 2800.00
        eth_pred_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=pred_horizon_test, freq='D')
        dummy_predictions['Ethereum'] = pd.DataFrame(
            {'predicted_close_price': np.linspace(2700, 2600, pred_horizon_test)},
            index=eth_pred_dates
        )
        dummy_volatilities['Ethereum'] = 0.85
        dummy_anomaly_statuses['Ethereum'] = True
    if len(cryptos_to_manage_test) > 2 and cryptos_to_manage_test[2] not in dummy_latest_prices:
        third_crypto = cryptos_to_manage_test[2]
        dummy_latest_prices[third_crypto] = 150.00
        third_pred_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=pred_horizon_test, freq='D')
        dummy_predictions[third_crypto] = pd.DataFrame(
             {'predicted_close_price': np.linspace(155, 165, pred_horizon_test)},
            index=third_pred_dates
        )
        dummy_volatilities[third_crypto] = 0.70
        dummy_anomaly_statuses[third_crypto] = False
    print("Initial Portfolio Summary:")
    print(optimizer_instance.get_current_portfolio_summary(dummy_latest_prices))
    target_allocs_dict = optimizer_instance.generate_target_percentage_allocations(
        dummy_predictions, dummy_volatilities, dummy_latest_prices, dummy_anomaly_statuses
    )
    if target_allocs_dict:
        simulated_trades_list = optimizer_instance.simulate_rebalance_to_targets(
            target_allocs_dict, dummy_latest_prices
        )
        if simulated_trades_list:
            print("\n--- Simulated Trades Log ---")
            for trade in simulated_trades_list:
                print(trade)
        else:
            print("\nNo rebalancing trades were simulated based on current targets.")
    print("\nFinal Portfolio Summary (after simulated rebalance attempt):")
    print(optimizer_instance.get_current_portfolio_summary(dummy_latest_prices))