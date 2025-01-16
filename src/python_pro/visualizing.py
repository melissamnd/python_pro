import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
import os

#---------------------------------------------------------
# Graphic analysis of the generated backtest file.
#---------------------------------------------------------

class PortfolioVisualizer:
    def __init__(self, data=None):
        """
        Constructor to initialize the PortfolioVisualizer class.

        Parameters:
        - data: Optional DataFrame containing asset data (prices, weights, etc.).
        """
        self.data = data

    def plot_historical_prices(self, df):
        """Plots historical prices from a DataFrame."""
        
        # Vérifiez si les données sont disponibles et non vides
        if df.empty:
            logging.warning("No data available to plot historical prices.")
            return
        
        tickers = df['ticker'].unique()  # Récupère tous les tickers uniques
        plt.figure(figsize=(12, 6))

        # Plot pour chaque ticker dans l'univers
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]
            
            if not ticker_data.empty:  # Vérifiez si des données existent pour ce ticker
                plt.plot(ticker_data['Date'], ticker_data['Adj Close'], label=ticker)
            else:
                logging.warning(f"No data available for ticker {ticker}.")

        # Ajouter le titre et les étiquettes
        plt.title("Historical Prices of Selected Stocks")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price ($)")
        plt.legend()

        # Affichez le graphique
        plt.show()

        # Sauvegardez dans le répertoire 'backtests_graphs' si nécessaire
        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')

        plt.savefig(f"backtests_graphs/Historical_Prices_{self.data['backtest_name']}.png", dpi=900)
        plt.show()

    def plot_var(self):
        """Plot Value at Risk (VaR) over time"""
        returns = self.calculate_returns()
        var = [self.calculate_var(confidence_level=0.95)] * len(returns)  # VaR constant over time for simplicity

        # Create a plot of VaR
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(returns)), var, label="VaR (95% confidence)", color="red")
        plt.title("Value at Risk (VaR) Over Time")
        plt.xlabel("Time")
        plt.ylabel("VaR")
        plt.legend()

        # Save the figure to the 'backtests_graphs' folder
        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')
        
        plt.savefig(f"backtests_graphs/Value_at_Risk_{self.data['initial_value']}_{self.data['final_value']}.png", dpi=900)
        plt.show()

    def calculate_returns(self):
        """Calculates the returns of the portfolio"""
        return np.diff(self.data['PortfolioValue']) / self.data['PortfolioValue'][:-1]

    def calculate_var(self, confidence_level=0.95):
        """Calculates the Value at Risk (VaR)"""
        returns = self.calculate_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

def analyze_all_transactions(backtest_instance):
    """
    Analyse toutes les transactions (achats et ventes) pour chaque action présente
    dans le fichier généré par le backtest.
    """
    # Récupérer le nom du fichier de backtest généré
    backtest_name = backtest_instance.backtest_name  # Nom généré dans l'instance de Backtest

    # Définir le chemin complet du fichier CSV
    transaction_log_path = f'backtests/{backtest_name}.csv'
    
    # Vérifier si le fichier existe avant de le lire
    if os.path.exists(transaction_log_path):
        # Lire le fichier CSV du backtest
        df = pd.read_csv(transaction_log_path)
        
        # Créer le dossier backtest_stats s'il n'existe pas déjà
        if not os.path.exists('backtest_stats'):
            os.makedirs('backtest_stats')

        stats_list = []  # Liste pour stocker les résultats à sauvegarder dans un fichier
        
        # Récupérer tous les tickers uniques
        tickers = df['Ticker'].unique()
        print(f"Unique tickers found: {tickers}")

        # Analyser chaque ticker
        for ticker in tickers:
            print(f"\nAnalyzing transactions for {ticker}:", flush=True)
            
            # Filtrer les transactions pour ce ticker
            ticker_df = df[df['Ticker'] == ticker]
            
            # Filtrer les achats et les ventes
            buy_ticker = ticker_df[ticker_df['Action'] == 'BUY']
            sell_ticker = ticker_df[ticker_df['Action'] == 'SELL']

            # Calculer la quantité totale achetée et vendue
            total_bought = buy_ticker['Quantity'].sum()
            total_sold = sell_ticker['Quantity'].sum()

            # Calculer le prix moyen d'achat et de vente
            avg_buy_price = (buy_ticker['Quantity'] * buy_ticker['Price']).sum() / total_bought if total_bought > 0 else 0
            avg_sell_price = (sell_ticker['Quantity'] * sell_ticker['Price']).sum() / total_sold if total_sold > 0 else 0

            # Affichage des résultats pour ce ticker
            print(f"Total {ticker} bought: {total_bought} shares", flush=True)
            print(f"Total {ticker} sold: {total_sold} shares", flush=True)
            print(f"Average buy price for {ticker}: ${avg_buy_price:.2f}", flush=True)
            print(f"Average sell price for {ticker}: ${avg_sell_price:.2f}", flush=True)

            # Ajouter les résultats à la liste stats_list pour les sauvegarder dans un fichier
            stats_list.append({
                "Ticker": ticker,
                "Total Bought": total_bought,
                "Total Sold": total_sold,
                "Average Buy Price": avg_buy_price,
                "Average Sell Price": avg_sell_price
            })
        
        # Convertir les résultats en DataFrame et les enregistrer dans un fichier CSV
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv('backtest_stats/transaction_analysis.csv', index=False)
        print(stats_df)
        print(f"Transaction analysis saved to 'backtest_stats/transaction_analysis.csv'", flush=True)

    else:
        print(f"Le fichier de backtest {transaction_log_path} n'existe pas.", flush=True)
        return None
