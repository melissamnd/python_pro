### MY NEW MODULE FUNCTIONS MODIFYING EXISTING FUNCTIONS AND ADDING NEW FUNCTIONS
### ONLY USER INTERACTIVE FUNCTIONS TO GET ALL DYNAMIC INPUTS

import tkinter as tk
import pandas as pd
import pybacktestchain
import numpy as np
import logging
import yfinance as yf
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from IPython.display import display, Markdown
from tkinter import simpledialog, messagebox
from pybacktestchain.data_module import get_stocks_data, get_stock_data  # Import the existing functions

class Data_module2():
    @staticmethod
    def get_stock_data(ticker, start_date, end_date):
        """Retrieve historical stock data for a single ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
            data['ticker'] = ticker
            data.reset_index(inplace=True)
            return data[['Date', 'ticker', 'Adj Close', 'Volume']]
        except Exception as e:
            logging.warning(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_stocks_data(tickers, start_date, end_date):
        """Retrieve historical stock data for multiple tickers."""
        dfs = []
        
        for ticker in tickers:
            df = Data_module2.get_stock_data(ticker, start_date, end_date)
            if not df.empty:
                df = df[['Date', 'Adj Close']]  # Keep only 'Date' and 'Adj Close'
                df['Ticker'] = ticker  # Add a column for ticker
                dfs.append(df)
        
        # Combine all dataframes into one
        all_data = pd.concat(dfs)
        
        # Pivot the data so tickers are columns and dates are rows
        all_data = all_data.pivot(index='Date', columns='Ticker', values='Adj Close')
        
        # Drop rows with NaN values
        all_data = all_data.dropna(how='all')
        
        return all_data


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get stock inputs
def get_stock_inputs():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    num_stocks = simpledialog.askinteger("Number of Stocks", "How many stocks do you want to enter?")
    
    if not num_stocks or num_stocks <= 0:
        print("No stocks to enter.")
        return []

    root = tk.Tk()
    root.title("Enter Stock Names")
    entries = []
    stock_names = []

    def on_submit():
        nonlocal stock_names
        stock_names = [entry.get().strip().upper() for entry in entries]
        root.quit()
        root.destroy()

    for i in range(num_stocks):
        tk.Label(root, text=f"Stock {i+1}").pack(padx=10, pady=5)
        entry = tk.Entry(root)
        entry.pack(padx=10, pady=5)
        entries.append(entry)

    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    root.mainloop()
    return stock_names

# Function to get date inputs via a userform
def get_date_inputs():
    # Fonction pour demander la saisie des dates via une input box (fenêtre graphique)
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale de Tkinter

    # Demander la date de début
    start_date_str = simpledialog.askstring("Start Date", "Please enter the start date (YYYY-MM-DD):")
    if not start_date_str:
        return None, None  # Si l'utilisateur annule ou ne rentre rien, on retourne None

    try:
        # Valider et convertir la date de début
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Invalid Date", "Invalid start date format. Please use YYYY-MM-DD.")
        return None, None  # Retourner None en cas d'erreur

    # Demander la date de fin
    end_date_str = simpledialog.askstring("End Date", "Please enter the end date (YYYY-MM-DD):")
    if not end_date_str:
        return None, None  # Si l'utilisateur annule ou ne rentre rien, on retourne None

    try:
        # Valider et convertir la date de fin
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Invalid Date", "Invalid end date format. Please use YYYY-MM-DD.")
        return None, None  # Retourner None en cas d'erreur

    if start_date > end_date:
        messagebox.showerror("Invalid Date Range", "Start date must be before end date!")
        return None, None  # Retourner None en cas d'erreur
    
    return start_date, end_date
    
def get_target_return():
    """
    Prompt the user to input a target return using a simple dialog box.
    Returns the target return as a float.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    target_return = simpledialog.askfloat(
        "Target Return", 
        "Enter the target return (e.g., 0.1 for 10%):",
        minvalue=0.0, maxvalue=1.0  # Optional bounds for input
    )
    root.destroy()  # Close the Tkinter window
    if target_return is None:
        raise ValueError("No target return provided!")
    return target_return

# Function to get the rebalancing strategy input from the user
def get_rebalancing_strategy():
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale de Tkinter
    
    # Demander la stratégie de rééquilibrage
    choices = ["End of Week", "End of Month", "Every Quarter"]
    
    rebalancing_strategy = simpledialog.askstring("Rebalancing Strategy", 
                                                  "Choose rebalancing strategy: End of Week, End of Month, or Every Quarter")
    
    # Valider le choix
    if rebalancing_strategy not in choices:
        print("Invalid choice. Please choose one of the following strategies: End of Week, End of Month, or Every Quarter.")
        return None
    
    # Retourner la classe appropriée en fonction du choix
    if rebalancing_strategy == "End of Week":
        return EndOfWeek  # Retourne la classe, pas la chaîne de caractères
    elif rebalancing_strategy == "End of Month":
        return EndOfMonth
    elif rebalancing_strategy == "Every Quarter":
        return EveryQuarter
        
def get_stop_loss_threshold():
    def on_submit():
        nonlocal stop_loss_threshold
        try:
            user_input = threshold_entry.get()
            # Vérifier que l'entrée n'est pas vide
            if not user_input:
                raise ValueError("Input cannot be empty.")
            
            # Convertir l'entrée en float
            stop_loss_threshold = float(user_input)
            
            # Vérifier que la valeur est positive
            if stop_loss_threshold <= 0:
                raise ValueError("Threshold must be a positive number.")
            
            root.quit()  # Fermer la fenêtre de l'input
            root.destroy()  # Détruire la fenêtre
        except ValueError as e:
            # Afficher un message d'erreur si l'entrée est invalide
            messagebox.showerror("Invalid Input", f"Invalid input: {e}")
    
    # Initialisation de la variable stop_loss_threshold
    stop_loss_threshold = None
    
    # Créer la fenêtre Tkinter
    root = tk.Tk()
    root.title("Enter Stop-Loss Threshold")
    
    # Ajouter un label et un champ de saisie
    tk.Label(root, text="Enter Stop-Loss Threshold (e.g., 0.1 for 10%):").pack(padx=10, pady=5)
    threshold_entry = tk.Entry(root)
    threshold_entry.pack(padx=10, pady=5)
    
    # Ajouter un bouton de soumission
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    
    # Lancer la boucle principale de Tkinter
    root.mainloop()
    
    return stop_loss_threshold
    
def get_initial_cash_input():
    def on_submit():
        nonlocal initial_cash
        try:
            initial_cash = float(initial_cash_entry.get())
            if initial_cash <= 0:
                raise ValueError("Initial cash must be a positive number.")
            root.quit()
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid input: {e}")
    
    initial_cash = None
    
    root = tk.Tk()
    root.title("Enter Initial Cash")
    
    tk.Label(root, text="Enter Initial Cash Amount:").pack(padx=10, pady=5)
    initial_cash_entry = tk.Entry(root)
    initial_cash_entry.pack(padx=10, pady=5)
    
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    root.mainloop()
    
    return initial_cash
