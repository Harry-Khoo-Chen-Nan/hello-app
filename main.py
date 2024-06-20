import streamlit as st
import os
import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt

def read_csv_files_from_folder(folder_path):
    df_dictionary = {}

   # Construct the full path to the folder containing Excel files
    folder_path = os.path.join(os.path.dirname(__file__), folder_path)

    # List all entries in the specified directory
    for filename in os.listdir(folder_path): # listdir() returns a list ['mean_reversal.csv', 'fxpmo.csv', 'momentum_day_trading.csv']
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename) # concat the 2 strings with /
            df = pd.read_csv(file_path)
            filename_without_extension = os.path.splitext(filename)[0]
            df_dictionary[filename_without_extension] = df

    return df_dictionary

def main():

    df_dictionary = read_csv_files_from_folder(folder_path = 'excel_files') # key = str, value = df
    list_dictionary = {} # key = str, value = list (converted from df)

    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>Trade Analytics</h1>", unsafe_allow_html = True)

    start_date = pd.to_datetime('2023-06-14')
    end_date = pd.to_datetime('2024-06-14')
    new_dates = pd.date_range(start = start_date, end = end_date, freq = 'D')

    # Data imputation and cleaning
    for key, value in df_dictionary.items():
        list_dictionary[key] = value.values.tolist()
        df_dictionary[key].drop('Balance', axis = 1, inplace = True)
        df_dictionary[key]['Percentage Change'] = df_dictionary[key]['Equity'].pct_change() * 100
        df_dictionary[key].loc[0, 'Percentage Change'] = 0.0
        df_dictionary[key]['Date'] = pd.to_datetime(df_dictionary[key]['Date'])

        df_dictionary[key].set_index('Date', inplace = True)
        df_dictionary[key] = df_dictionary[key].reindex(new_dates)
        df_dictionary[key]['Percentage Change'].fillna(0, inplace = True)
        df_dictionary[key]['Equity'].fillna(0, inplace = True)
        df_dictionary[key] = df_dictionary[key].reset_index().rename(columns = {"index": "Date"})
        # st.write(key)
        # st.write(df_dictionary[key])

    portfolio_df = pd.DataFrame({'Date': df_dictionary['fxpmo']['Date'], 'mean_reversal': df_dictionary['mean_reversal']['Percentage Change'], 'momentum_day_trading': df_dictionary['momentum_day_trading']['Percentage Change'], 'fxpmo': df_dictionary['fxpmo']['Percentage Change']})
    portfolio_df.set_index('Date', inplace = True)
    Y = portfolio_df.copy()
    port = rp.Portfolio(returns=Y)
    method_mu='hist'
    method_cov='hist'
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    left_col, right_col = st.columns(2)

    with left_col:
        
        risk_measure_col, obj_col = st.columns(2)

        with risk_measure_col:
            risk_measure_options = ['MV', 'MAD', 'GMD', 'MSV', 'FLPM', 'SLPM', 'CVaR', 'TG', 'EVaR', 'RLVaR', 'WR', 'RG', 'CVRG', 'TGRG', 'MDD', 'ADD']
            rm = st.selectbox("Risk Measure", risk_measure_options, index = 1)
        with obj_col:
            objective_functions_options = ['MinRisk','Sharpe']
            obj = st.selectbox("Objective", objective_functions_options, index = 0) # objective function.  E.g. MinRisk, MaxRet, Utility or Sharpe

        # Estimate optimal portfolio:
        model='Classic' # Classic (historical), BL (Black Litterman) or FM (Factor Model)
        hist = True
        rf = 0 # Risk free rate
        l = 0 # Risk aversion factor
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        # st.write(w.T)
        percentages = w.T.iloc[0] * 100
        labels = percentages.index
        sizes = percentages.values

        # Create a donut chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
        wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.3), startangle=90, colors=plt.get_cmap("tab20").colors, textprops=dict(color="w"))

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(f"{labels[i]}: {sizes[i]:.2f}%", xy=(x, y), xytext=(1.5*np.sign(x), 1.5*y),
                        horizontalalignment=horizontalalignment, **kw)

        # Add a legend
        ax.legend(wedges, labels, title="Strategies", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax.set_title('Portfolio Weightages')

        # Display the pie chart in Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()