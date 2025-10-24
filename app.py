#!/usr/bin/env python
# coding: utf-8

# In[9]:


import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "COT Analysis Dashboard"

# Load data
df = pd.read_csv('Data/Disaggregated_Futures_Only.csv')
df['Report_Date'] = pd.to_datetime(df['Report_Date_as_YYYY_MM_DD'], format='mixed')

# Define specific Contract_Market_Name filtering for commodities with multiple contract types
def filter_commodity_contracts(df):
    """
    Filter the dataframe to keep only specific contract types for certain commodities
    to avoid duplicate data for the same date.
    """
    # Define the specific contract market names for each commodity
    commodity_filters = {
        'NATURAL GAS': 'NAT GAS NYME',
        'GOLD': 'GOLD',
        'GASOLINE': 'GASOLINE RBOB',
        'SOYBEANS': 'SOYBEANS',
        'WHEAT': 'WHEAT-SRW'
    }
    
    # Create a mask for rows to keep
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Apply filters for each commodity
    for commodity_name, contract_market_name in commodity_filters.items():
        # Find rows for this commodity
        commodity_mask = df['Commodity Name'] == commodity_name
        # For this commodity, only keep rows with the specific Contract_Market_Name
        if commodity_mask.any():
            mask = mask & (~commodity_mask | (df['Contract_Market_Name'] == contract_market_name))
    
    return df[mask]

# Apply the filtering
df = filter_commodity_contracts(df)

# Load and process the price data file
try:
    # 1. Load the CSV
    price_df = pd.read_csv('Data/commodity_prices.csv')

    # 2. Reshape data from wide to long format using pandas.melt
    price_df = pd.melt(price_df, id_vars=['Date'], var_name='Commodity Name', value_name='Price')

    # 3. Convert the 'Date' column using the specific MM/DD/YYYY format
    price_df['Date'] = pd.to_datetime(price_df['Date'], format='mixed')

    # 4. Convert commodity names to uppercase to match the COT data
    price_df['Commodity Name'] = price_df['Commodity Name'].str.upper()

    # 5. Filter for Tuesdays to align with COT Report_Date
    price_df = price_df[price_df['Date'].dt.dayofweek == 1]

except Exception as e:
    print(f"Warning: Could not load or process 'Data/commodity_prices.csv'. Price features will be disabled. Error: {e}")
    price_df = pd.DataFrame() # Create empty dataframe to avoid errors


# Get unique commodities and report weeks for dropdowns
commodities = sorted(df['Commodity Name'].unique())
report_weeks = sorted(df['YYYY Report Week WW'].unique(), reverse=True)

# Get date range for the date picker
min_date = df['Report_Date'].min()
max_date = df['Report_Date'].max()

# App layout with tabs
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("COT Analysis Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'})
    ]),

    # Tabs for different visualizations
    dcc.Tabs(id='main-tabs', value='positioning-tab', children=[
        dcc.Tab(label='Positioning Analysis', value='positioning-tab'),
        dcc.Tab(label='Clustering, Concentration and Position Size 1', value='clustering1-tab'),
        dcc.Tab(label='Clustering, Concentration and Position Size 2', value='clustering2-tab'),
        dcc.Tab(label='Commodity Baskets', value='baskets-tab'),
        dcc.Tab(label='Position Changes Over Time', value='position-changes-tab'),
        dcc.Tab(label='Regression vs. Price', value='regression-tab'),
    ], style={'marginBottom': 20}),

    # Tab content area
    html.Div(id='tab-content')

], style={'maxWidth': '1500px', 'margin': '0 auto', 'padding': '20px'})


# Callback for rendering tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(active_tab):
    if active_tab == 'positioning-tab':
        return render_positioning_tab()
    elif active_tab == 'clustering1-tab':
        return render_clustering1_tab()
    elif active_tab == 'clustering2-tab':
        return render_clustering2_tab()
    elif active_tab == 'baskets-tab':
        return render_baskets_tab()
    elif active_tab == 'position-changes-tab':
        return render_position_changes_tab()
    elif active_tab == 'regression-tab':
        return render_regression_tab()
    else:
        return html.Div('Select a tab to view content')


def render_position_changes_tab():
    """Render the position changes over time tab"""
    trader_categories_options = [
        {'label': 'Managed Money', 'value': 'M_Money'},
        {'label': 'Producer/Merchant', 'value': 'Prod_Merc'},
        {'label': 'Swap Dealers', 'value': 'Swap'},
        {'label': 'Other Reportables', 'value': 'Other_Rept'},
        {'label': 'Non-Reportables', 'value': 'NonRept'}
    ]

    def create_line_selector(line_num):
        return html.Div([
            html.Label(f'Line {line_num} Categories (Summed):', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id=f'position-changes-categories-dropdown-{line_num}',
                options=trader_categories_options,
                value=['M_Money'] if line_num == 1 else [],
                multi=True,
                style={'width': '100%'}
            )
        ], style={'width': '19%', 'display': 'inline-block', 'margin': '0 0.5%'})

    return html.Div([
        # Control Panel
        html.Div([
            # First row - Commodity, Date Range, Display Type
            html.Div([
                html.Div([
                    html.Label('Select Commodity:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='position-changes-commodity-dropdown',
                        options=[{'label': c, 'value': c} for c in commodities],
                        value='LIVE CATTLE',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '28%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Select Date Range:', style={'fontWeight': 'bold'}),
                    dcc.DatePickerRange(
                        id='position-changes-date-range',
                        start_date=max_date - timedelta(days=365*2),
                        end_date=max_date,
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ], style={'width': '28%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Display Options:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='position-changes-display-type',
                        options=[
                            {'label': '  Net Positions', 'value': 'net'},
                            {'label': '  Long Positions', 'value': 'long'},
                            {'label': '  Short Positions', 'value': 'short'},
                            {'label': '  Percentage Change', 'value': 'percent'}
                        ],
                        value='net',
                        inline=True,
                        style={'marginTop': '10px'}
                    )
                ], style={'width': '28%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Chart Options:', style={'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id='position-changes-chart-options',
                        options=[
                            {'label': '  Show Price', 'value': 'show_price'},
                            {'label': '  Show Historical Max', 'value': 'show_hist_max'},
                            {'label': '  Show Historical Min', 'value': 'show_hist_min'}
                        ],
                        value=['show_price'] if not price_df.empty else [],
                        inline=False,
                        style={'marginTop': '10px'}
                    )
                ], style={'width': '12%', 'display': 'inline-block', 'marginLeft': '2%'})

            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            # Second row - Line Selectors
            html.Div([
                create_line_selector(1),
                create_line_selector(2),
                create_line_selector(3),
                create_line_selector(4),
                create_line_selector(5),
            ], style={'marginTop': '20px', 'display': 'flex'})

        ], style={'marginBottom': 30, 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        dcc.Graph(id='position-changes-chart', style={'height': '600px'}),

        html.Div([
            html.H3('Period Statistics', style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
            html.Div(id='position-changes-stats', style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})
    ])


def render_positioning_tab():
    """Render the positioning analysis tab"""
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Select Commodity:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='commodity-dropdown',
                        options=[{'label': c, 'value': c} for c in commodities],
                        value='LIVE CATTLE',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Select Trader Group:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='subgroup-dropdown',
                        options=[
                            {'label': 'Managed Money', 'value': 'M_Money'},
                            {'label': 'Producer/Merchant', 'value': 'Prod_Merc'},
                            {'label': 'Swap Dealers', 'value': 'Swap_Dealer'},
                            {'label': 'Other Reportables', 'value': 'Other_Rept'}
                        ],
                        value='M_Money',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ]),

            html.Div([
                html.Label('Visualization Style:', style={'fontWeight': 'bold', 'marginTop': '15px'}),
                dcc.RadioItems(
                    id='viz-style',
                    options=[
                        {'label': '  Highlight Recent Points (Gradient)', 'value': 'recent'},
                        {'label': '  Color by Year', 'value': 'yearly'}
                    ],
                    value='recent',
                    inline=True,
                    style={'marginTop': '10px'}
                )
            ], style={'width': '100%', 'marginTop': '15px'})

        ], style={'marginBottom': 30, 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        html.Div([
            dcc.Graph(id='positioning-chart', style={'height': '650px'})
        ], style={'marginBottom': 20}),

        html.Div([
            html.H3('Position Statistics', style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.Div(id='stats-panel', style={'display': 'flex', 'justifyContent': 'space-around'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        html.Div([
            html.Div(id='date-range', style={'textAlign': 'center', 'marginTop': 20, 'color': '#7f8c8d'})
        ])
    ])


def render_clustering1_tab():
    """Render the first clustering analysis tab (3 separate line plots)"""
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Select Commodity:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='clustering1-commodity-dropdown',
                        options=[{'label': c, 'value': c} for c in commodities],
                        value='LIVE CATTLE',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Select Position Type:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='clustering1-position-dropdown',
                        options=[
                            {'label': 'Managed Money Long', 'value': 'M_Money_Positions_Long'},
                            {'label': 'Managed Money Short', 'value': 'M_Money_Positions_Short'},
                            {'label': 'Producer/Merchant Long', 'value': 'Prod_Merc_Positions_Long'},
                            {'label': 'Producer/Merchant Short', 'value': 'Prod_Merc_Positions_Short'},
                            {'label': 'Swap Dealers Long', 'value': 'Swap_Positions_Long'},
                            {'label': 'Swap Dealers Short', 'value': 'Swap_Positions_Short'},
                        ],
                        value='M_Money_Positions_Long',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ])
        ], style={'marginBottom': 30, 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        html.Div([
            dcc.Graph(id='concentration-ratio-chart', style={'height': '400px'}),
            dcc.Graph(id='clustering-chart', style={'height': '400px'}),
            dcc.Graph(id='position-size-chart', style={'height': '400px'})
        ])
    ])


def render_clustering2_tab():
    """Render the second clustering analysis tab (scatter plot with color gradient)"""
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Select Commodity:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='clustering2-commodity-dropdown',
                        options=[{'label': c, 'value': c} for c in commodities],
                        value='LIVE CATTLE',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Select Position Type:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='clustering2-position-dropdown',
                        options=[
                            {'label': 'Managed Money Long', 'value': 'M_Money_Positions_Long'},
                            {'label': 'Managed Money Short', 'value': 'M_Money_Positions_Short'},
                            {'label': 'Producer/Merchant Long', 'value': 'Prod_Merc_Positions_Long'},
                            {'label': 'Producer/Merchant Short', 'value': 'Prod_Merc_Positions_Short'},
                            {'label': 'Swap Dealers Long', 'value': 'Swap_Positions_Long'},
                            {'label': 'Swap Dealers Short', 'value': 'Swap_Positions_Short'},
                        ],
                        value='M_Money_Positions_Long',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ])
        ], style={'marginBottom': 30, 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        html.Div([
            dcc.Graph(id='clustering2-scatter-chart', style={'height': '700px'})
        ])
    ])


def render_baskets_tab():
    """Render the commodity baskets tab (Treemap or Stacked Bar)"""
    return html.Div([
        html.Div([
            html.Div([
                html.Label('Select Report Week:', style={'fontWeight': 'bold'}), 
                dcc.Dropdown(
                    id='week-dropdown',
                    options=[{'label': w, 'value': w} for w in report_weeks],
                    value=report_weeks[0] if report_weeks else None,
                    clearable=False, 
                    style={'width': '48%'}
                )
            ]),
            html.Div([
                html.Label('Select Commodities (leave empty for all):', style={'fontWeight': 'bold', 'marginTop': '15px'}), 
                dcc.Dropdown(
                    id='commodities-filter-dropdown',
                    options=[{'label': c, 'value': c} for c in commodities],
                    value=['GOLD', 'CRUDE OIL', 'CORN', 'SOYBEANS', 'SUGAR', 
                           'WHEAT', 'COTTON', 'LEAN HOGS', 'LIVE CATTLE', 'FEEDER CATTLE'], 
                    multi=True,
                    style={'width': '100%'}
                )
            ], style={'marginTop': '15px'}),
            html.Div([
                html.Label('Display View:', style={'fontWeight': 'bold', 'marginTop': '15px'}), 
                dcc.RadioItems(
                    id='treemap-display-mode',
                    options=[
                        {'label': '  Hierarchical Treemap', 'value': 'hierarchical'}, 
                        {'label': '  Stacked Bar View', 'value': 'stacked_bar'}
                    ],
                    value='hierarchical',
                    inline=True, 
                    style={'marginTop': '10px'}
                )
            ], style={'marginTop': '15px'})
        ], style={'marginBottom': 30, 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        html.Div([
             dcc.Graph(id='treemap-chart', style={'height': '800px'}) 
        ])
    ])


def render_regression_tab():
    """Render the regression analysis tab"""
    if price_df.empty:
        return html.Div([
            html.H3("Price Data Not Found or Incorrectly Formatted"),
            html.P("Please ensure 'Data/commodity_prices.csv' exists and is correctly formatted.")
        ], style={'textAlign': 'center', 'color': 'red', 'marginTop': 50})

    return html.Div([
        html.Div([
            html.Div([
                html.Label('Select Commodity:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='regression-commodity-dropdown',
                    options=[{'label': c, 'value': c} for c in commodities],
                    value='LIVE CATTLE',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label('Select Trader Category to Plot:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='regression-category-dropdown',
                    options=[
                        {'label': 'Managed Money', 'value': 'M_Money'},
                        {'label': 'Producer/Merchant', 'value': 'Prod_Merc'},
                        {'label': 'Swap Dealers', 'value': 'Swap'},
                        {'label': 'Other Reportables', 'value': 'Other_Rept'},
                        {'label': 'Non-Reportables', 'value': 'NonRept'}
                    ],
                    value='M_Money',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
        ], style={'marginBottom': 30, 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='regression-chart')
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H4('Price Correlation Summary', style={'textAlign': 'center'}),
                html.Div(id='regression-summary-table')
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%', 'marginTop': '50px'})
        ])
    ])


# --- Callbacks ---

@app.callback(
    [Output('position-changes-chart', 'figure'),
     Output('position-changes-stats', 'children')],
    [Input('position-changes-commodity-dropdown', 'value'),
     Input('position-changes-date-range', 'start_date'),
     Input('position-changes-date-range', 'end_date'),
     Input('position-changes-display-type', 'value'),
     Input('position-changes-chart-options', 'value')] +
    [Input(f'position-changes-categories-dropdown-{i}', 'value') for i in range(1, 6)]
)
def update_position_changes_chart(commodity, start_date, end_date, display_type, chart_options, line1_cats, line2_cats, line3_cats, line4_cats, line5_cats):
    """Update the position changes chart, with optional price overlay."""
    lines_to_plot = [line1_cats, line2_cats, line3_cats, line4_cats, line5_cats]
    show_price = 'show_price' in chart_options if chart_options else False
    show_hist_max = 'show_hist_max' in chart_options if chart_options else False
    show_hist_min = 'show_hist_min' in chart_options if chart_options else False

    if not commodity:
        return go.Figure(), []

    # Filter main dataframe by commodity and date
    df_filtered = df[(df['Commodity Name'] == commodity) &
                     (df['Report_Date'] >= start_date) & 
                     (df['Report_Date'] <= end_date)].copy()
    df_filtered.sort_values('Report_Date', inplace=True)

    # Get historical data for max calculation (all dates for this commodity)
    df_historical = df[df['Commodity Name'] == commodity].copy()
    df_historical.sort_values('Report_Date', inplace=True)

    # If showing price, merge with price data
    if show_price and not price_df.empty:
        price_comm_df = price_df[price_df['Commodity Name'] == commodity]
        df_filtered = pd.merge(df_filtered, price_comm_df[['Date', 'Price']], left_on='Report_Date', right_on='Date', how='left')

    if df_filtered.empty:
        return go.Figure(layout={'title': 'No data for selected range'}), []

    # Initialize figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    stats_data = []

    # Helper function to lighten a color
    def lighten_color(color_str, amount=0.5):
        """Lighten a hex or named color by blending with white"""
        # Handle named colors by converting to a lighter version
        named_colors = {
            '#636efa': '#a6aff5',  # blue
            '#EF553B': '#f79a8a',  # red
            '#00cc96': '#7fe5c9',  # teal
            '#ab63fa': '#d5a6fc',  # purple
            '#FFA15A': '#ffc999',  # orange
            '#19d3f3': '#8ce9f9',  # cyan
            '#FF6692': '#ffb0c8',  # pink
            '#B6E880': '#daf3bf',  # lime
            '#FF97FF': '#ffcbff',  # magenta
            '#FECB52': '#fee498',  # yellow
        }
        
        # Check if it's a named Plotly color
        if color_str in named_colors:
            return named_colors[color_str]
        
        # Handle hex colors
        try:
            if color_str.startswith('#'):
                color_str = color_str[1:]
            if len(color_str) == 6:
                r, g, b = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
                r = int(r + (255 - r) * amount)
                g = int(g + (255 - g) * amount)
                b = int(b + (255 - b) * amount)
                return f'#{r:02x}{g:02x}{b:02x}'
        except:
            pass
        
        return color_str

    if any(lines_to_plot):
        label_map = {'M_Money': 'Managed Money', 'Prod_Merc': 'Producer/Merchant', 'Swap': 'Swap Dealers',
                     'Other_Rept': 'Other Reportables', 'NonRept': 'Non-Reportables'}

        # Calculate net positions for both filtered and historical data
        for cat_code in label_map.keys():
            long_col, short_col = (f'Swap_Positions_Long_All', f'Swap_Positions_Short_All') if cat_code == 'Swap' else (f'{cat_code}_Positions_Long_All', f'{cat_code}_Positions_Short_All')
            
            # For filtered data
            if long_col in df_filtered.columns and short_col in df_filtered.columns:
                df_filtered[f'{cat_code}_net'] = df_filtered[long_col] - df_filtered[short_col]
                df_filtered[f'{cat_code}_long'] = df_filtered[long_col]
                df_filtered[f'{cat_code}_short'] = df_filtered[short_col]
                net_pos_series = df_filtered[f'{cat_code}_net']
                if not net_pos_series.empty and net_pos_series.iloc[0] != 0:
                    df_filtered[f'{cat_code}_percent'] = (net_pos_series - net_pos_series.iloc[0]) / abs(net_pos_series.iloc[0]) * 100
                else:
                    df_filtered[f'{cat_code}_percent'] = 0
            
            # For historical data
            if long_col in df_historical.columns and short_col in df_historical.columns:
                df_historical[f'{cat_code}_net'] = df_historical[long_col] - df_historical[short_col]

        # Get default plotly colors
        default_colors = px.colors.qualitative.Plotly
        
        for line_idx, categories in enumerate(lines_to_plot):
            if not categories: continue
            
            # Calculate line values for filtered data
            line_values = pd.Series(0, index=df_filtered.index)
            for category in categories:
                line_values += df_filtered.get(f'{category}_{display_type}', 0)

            line_label = ' + '.join([label_map.get(c, c) for c in categories])
            
            # Get the color for this line
            line_color = default_colors[line_idx % len(default_colors)]
            
            # Add main line trace
            fig.add_trace(go.Scatter(
                x=df_filtered['Report_Date'], 
                y=line_values, 
                name=line_label, 
                mode='lines',
                line=dict(color=line_color)
            ), secondary_y=False)

            # Calculate and add max/min historical net position lines (only for 'net' display type)
            if display_type == 'net' and (show_hist_max or show_hist_min):
                # Calculate historical net for these categories
                historical_net = pd.Series(0, index=df_historical.index)
                for category in categories:
                    historical_net += df_historical.get(f'{category}_net', 0)
                
                # Create lighter shades of the line color (max slightly lighter, min more lighter)
                lighter_color_max = lighten_color(line_color, amount=0.5)
                lighter_color_min = lighten_color(line_color, amount=0.6)
                
                # Add max historical line if requested
                if show_hist_max:
                    max_value = historical_net.max()
                    fig.add_trace(go.Scatter(
                        x=[df_filtered['Report_Date'].min(), df_filtered['Report_Date'].max()],
                        y=[max_value, max_value],
                        name=f'{line_label} (Max Historical)',
                        mode='lines',
                        line=dict(color=lighter_color_max, dash='dash', width=2),
                        showlegend=True
                    ), secondary_y=False)
                
                # Add min historical line if requested
                if show_hist_min:
                    min_value = historical_net.min()
                    fig.add_trace(go.Scatter(
                        x=[df_filtered['Report_Date'].min(), df_filtered['Report_Date'].max()],
                        y=[min_value, min_value],
                        name=f'{line_label} (Min Historical)',
                        mode='lines',
                        line=dict(color=lighter_color_min, dash='dash', width=2),
                        showlegend=True
                    ), secondary_y=False)

            if not line_values.empty:
                current_val, start_val = line_values.iloc[-1], line_values.iloc[0]
                change = current_val - start_val
                pct_change = (change / abs(start_val)) * 100 if start_val != 0 else 0
                
                # Calculate historical values based on display type
                historical_values = pd.Series(0, index=df_historical.index)
                for category in categories:
                    historical_values += df_historical.get(f'{category}_{display_type}', 0)
                
                hist_max = historical_values.max() if not historical_values.empty else 0
                hist_min = historical_values.min() if not historical_values.empty else 0
                
                # Calculate percentile rank
                if len(historical_values) > 0 and not historical_values.empty:
                    percentile = (historical_values <= current_val).sum() / len(historical_values) * 100
                else:
                    percentile = 0
                
                # Calculate rate of change metrics
                if len(line_values) > 1:
                    # Average change per week over the selected period
                    avg_change = change / (len(line_values) - 1) if len(line_values) > 1 else 0
                    # Current change (last week vs previous week)
                    current_weekly_change = line_values.iloc[-1] - line_values.iloc[-2]
                    # Weeks to max/min at current rate
                    if current_weekly_change != 0:
                        weeks_to_max = abs((hist_max - current_val) / current_weekly_change)
                        weeks_to_min = abs((hist_min - current_val) / current_weekly_change)
                    else:
                        weeks_to_max = float('inf')
                        weeks_to_min = float('inf')
                else:
                    avg_change = 0
                    current_weekly_change = 0
                    weeks_to_max = float('inf')
                    weeks_to_min = float('inf')
                
                # Calculate seasonality data (only for net positions)
                seasonality_data = {}
                if display_type == 'net':
                    # Get current week of year
                    current_date = df_filtered['Report_Date'].iloc[-1]
                    current_week_of_year = current_date.isocalendar()[1]
                    current_year = current_date.year
                    
                    # Calculate seasonal averages for 5, 10, 20 years and 1, 2, 4 weeks forward
                    for years_back in [5, 10, 20]:
                        seasonality_data[years_back] = {}
                        for weeks_forward in [1, 2, 4]:
                            changes = []
                            # Look at historical data for the same week in past years
                            for year_offset in range(1, years_back + 1):
                                target_year = current_year - year_offset
                                # Find data points around this week in that year
                                year_data = df_historical[df_historical['Report_Date'].dt.year == target_year]
                                if len(year_data) > 0:
                                    # Find the closest week to our current week of year
                                    year_data = year_data.copy()
                                    year_data['week_of_year'] = year_data['Report_Date'].apply(lambda x: x.isocalendar()[1])
                                    week_diff = (year_data['week_of_year'] - current_week_of_year).abs()
                                    if week_diff.min() <= 2:  # Allow +/- 2 weeks tolerance
                                        base_idx = week_diff.idxmin()
                                        base_date = year_data.loc[base_idx, 'Report_Date']
                                        
                                        # Calculate net position at base date
                                        base_net = 0
                                        for category in categories:
                                            if f'{category}_net' in df_historical.columns:
                                                base_net += df_historical.loc[base_idx, f'{category}_net']
                                        
                                        # Find the date weeks_forward ahead
                                        target_date = base_date + pd.Timedelta(weeks=weeks_forward)
                                        future_data = df_historical[
                                            (df_historical['Report_Date'] >= target_date - pd.Timedelta(days=3)) &
                                            (df_historical['Report_Date'] <= target_date + pd.Timedelta(days=3))
                                        ]
                                        
                                        if len(future_data) > 0:
                                            future_idx = future_data.index[0]
                                            future_net = 0
                                            for category in categories:
                                                if f'{category}_net' in df_historical.columns:
                                                    future_net += df_historical.loc[future_idx, f'{category}_net']
                                            
                                            change_value = future_net - base_net
                                            changes.append(change_value)
                            
                            # Calculate average
                            seasonality_data[years_back][weeks_forward] = np.mean(changes) if len(changes) > 0 else 0
                else:
                    # If not net positions, populate with zeros or N/A
                    for years_back in [5, 10, 20]:
                        seasonality_data[years_back] = {1: 0, 2: 0, 4: 0}
                
                # Create the four-column layout
                stats_card = html.Div([
                    # Column 1: Current Position
                    html.Div([
                        html.H4('Current Position', style={'fontSize': '16px', 'marginBottom': '15px', 'borderBottom': '2px solid ' + line_color, 'paddingBottom': '5px'}),
                        html.Div([
                            html.P('Current:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{current_val:,.0f}', style={'fontSize': '18px', 'marginBottom': '10px'}),
                            html.P('Period Change:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{change:+,.0f}', style={'fontSize': '16px', 'color': '#2ecc71' if change > 0 else '#e74c3c', 'marginBottom': '10px'}),
                            html.P('% Change:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{pct_change:+.1f}%', style={'fontSize': '16px', 'color': '#2ecc71' if pct_change > 0 else '#e74c3c'})
                        ])
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    # Column 2: Historical Extremes
                    html.Div([
                        html.H4('Historical Context', style={'fontSize': '16px', 'marginBottom': '15px', 'borderBottom': '2px solid ' + line_color, 'paddingBottom': '5px'}),
                        html.Div([
                            html.P('Historical Max:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{hist_max:,.0f}', style={'fontSize': '16px', 'marginBottom': '10px'}),
                            html.P('Historical Min:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{hist_min:,.0f}', style={'fontSize': '16px', 'marginBottom': '10px'}),
                            html.P('Percentile Rank:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{percentile:.1f}%', style={'fontSize': '16px', 'color': '#2ecc71' if percentile > 50 else '#e74c3c'})
                        ])
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    # Column 3: Rate of Change
                    html.Div([
                        html.H4('Rate of Change', style={'fontSize': '16px', 'marginBottom': '15px', 'borderBottom': '2px solid ' + line_color, 'paddingBottom': '5px'}),
                        html.Div([
                            html.P('Avg Weekly Change:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{avg_change:+,.0f}', style={'fontSize': '16px', 'marginBottom': '10px'}),
                            html.P('Current Weekly Change:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(f'{current_weekly_change:+,.0f}', style={'fontSize': '16px', 'marginBottom': '10px'}),
                            html.P('Weeks to Max/Min:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                            html.P(
                                f"{weeks_to_max:.1f} / {weeks_to_min:.1f}" if weeks_to_max < 1000 and weeks_to_min < 1000 else "N/A",
                                style={'fontSize': '14px'}
                            )
                        ])
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    # Column 4: Seasonality Table
                    html.Div([
                        html.H4('Seasonality (Avg Forward Change)', style={'fontSize': '16px', 'marginBottom': '15px', 'borderBottom': '2px solid ' + line_color, 'paddingBottom': '5px'}),
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th('Period', style={'padding': '8px', 'fontSize': '12px', 'backgroundColor': '#e9ecef'}),
                                html.Th('5 Yr', style={'padding': '8px', 'fontSize': '12px', 'backgroundColor': '#e9ecef'}),
                                html.Th('10 Yr', style={'padding': '8px', 'fontSize': '12px', 'backgroundColor': '#e9ecef'}),
                                html.Th('20 Yr', style={'padding': '8px', 'fontSize': '12px', 'backgroundColor': '#e9ecef'})
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td('1 Week', style={'padding': '8px', 'fontSize': '11px', 'fontWeight': 'bold'}),
                                    html.Td(f"{seasonality_data[5][1]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'color': '#2ecc71' if seasonality_data[5][1] > 0 else '#e74c3c'}),
                                    html.Td(f"{seasonality_data[10][1]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'color': '#2ecc71' if seasonality_data[10][1] > 0 else '#e74c3c'}),
                                    html.Td(f"{seasonality_data[20][1]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'color': '#2ecc71' if seasonality_data[20][1] > 0 else '#e74c3c'})
                                ]),
                                html.Tr([
                                    html.Td('2 Week', style={'padding': '8px', 'fontSize': '11px', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                    html.Td(f"{seasonality_data[5][2]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'backgroundColor': '#f8f9fa', 'color': '#2ecc71' if seasonality_data[5][2] > 0 else '#e74c3c'}),
                                    html.Td(f"{seasonality_data[10][2]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'backgroundColor': '#f8f9fa', 'color': '#2ecc71' if seasonality_data[10][2] > 0 else '#e74c3c'}),
                                    html.Td(f"{seasonality_data[20][2]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'backgroundColor': '#f8f9fa', 'color': '#2ecc71' if seasonality_data[20][2] > 0 else '#e74c3c'})
                                ]),
                                html.Tr([
                                    html.Td('4 Week', style={'padding': '8px', 'fontSize': '11px', 'fontWeight': 'bold'}),
                                    html.Td(f"{seasonality_data[5][4]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'color': '#2ecc71' if seasonality_data[5][4] > 0 else '#e74c3c'}),
                                    html.Td(f"{seasonality_data[10][4]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'color': '#2ecc71' if seasonality_data[10][4] > 0 else '#e74c3c'}),
                                    html.Td(f"{seasonality_data[20][4]:+,.0f}", style={'padding': '8px', 'fontSize': '11px', 'color': '#2ecc71' if seasonality_data[20][4] > 0 else '#e74c3c'})
                                ])
                            ])
                        ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #dee2e6'})
                    ], style={'flex': '1.2', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'margin': '0 5px'})
                    
                ], style={'display': 'flex', 'marginBottom': '20px', 'border': '2px solid ' + line_color, 'borderRadius': '10px', 'padding': '10px', 'backgroundColor': 'white'})
                
                stats_data.append(stats_card)

    # Add Price Trace if selected
    if show_price and 'Price' in df_filtered.columns and not df_filtered['Price'].dropna().empty:
        fig.add_trace(go.Scatter(x=df_filtered['Report_Date'], y=df_filtered['Price'], name='Price',
                                 mode='lines', line=dict(color='gold', dash='dash')), secondary_y=True)

    y_label_map = {'net': 'Net Position', 'long': 'Long Positions', 'short': 'Short Positions', 'percent': 'Percentage Change (%)'}
    y_label = y_label_map.get(display_type, 'Value')

    fig.update_layout(
        title_text=f'{commodity} - Position Changes Over Time',
        hovermode='x unified', template='plotly_white', height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_yaxes(title_text=y_label, secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False)

    return fig, stats_data


# ### FIX: Corrected column name logic for 'Swap Dealer'
def get_positioning_columns(subgroup):
    """Handles inconsistent column naming for Swap Dealers in COT data."""
    if subgroup == 'Swap_Dealer':
        traders_long_col = 'Traders_Swap_Dealer_Long_All'
        traders_short_col = 'Traders_Swap_Dealer_Short_All'
        positions_long_col = 'Swap_Positions_Long_All'  # Note: No 'Dealer'
        positions_short_col = 'Swap_Positions_Short_All' # Note: No 'Dealer'
    else:
        traders_long_col = f'Traders_{subgroup}_Long_All'
        traders_short_col = f'Traders_{subgroup}_Short_All'
        positions_long_col = f'{subgroup}_Positions_Long_All'
        positions_short_col = f'{subgroup}_Positions_Short_All'
    return traders_long_col, traders_short_col, positions_long_col, positions_short_col

def create_recent_highlight_chart(df_filtered, commodity, subgroup):
    """Create chart with recent points highlighted using gradient"""
    traders_long_col, traders_short_col, positions_long_col, positions_short_col = get_positioning_columns(subgroup)

    # Prepare data
    x_long = df_filtered[traders_long_col].values
    y_long = df_filtered[positions_long_col].values
    x_short = df_filtered[traders_short_col].values
    y_short = -df_filtered[positions_short_col].values
    dates = df_filtered['Report_Date'].dt.strftime('%Y-%m-%d').values

    fig = go.Figure()

    # Add long positions (older points)
    fig.add_trace(go.Scatter(
        x=x_long[:-5], y=y_long[:-5], mode='markers', name='Long Positions',
        marker=dict(color='#1f77b4', size=8, opacity=0.5),
        text=[f"Date: {d}<br>Traders: {t:,.0f}<br>Position: {p:,.0f}" 
              for d, t, p in zip(dates[:-5], x_long[:-5], y_long[:-5])],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add short positions (older points)
    fig.add_trace(go.Scatter(
        x=x_short[:-5], y=y_short[:-5], mode='markers', name='Short Positions',
        marker=dict(color='#ff7f0e', size=8, opacity=0.5),
        text=[f"Date: {d}<br>Traders: {t:,.0f}<br>Position: {p:,.0f}" 
              for d, t, p in zip(dates[:-5], x_short[:-5], -y_short[:-5])],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add recent 5 points with gradient
    if len(x_long) > 5:
        colors_gradient = ['#666666', '#4d4d4d', '#333333', '#1a1a1a', '#000000']
        for i in range(5):
            idx = -5 + i
            fig.add_trace(go.Scatter(
                x=[x_long[idx]], y=[y_long[idx]], mode='markers',
                marker=dict(color=colors_gradient[i], size=12),
                text=f"Date: {dates[idx]}<br>Traders: {x_long[idx]:,.0f}<br>Position: {y_long[idx]:,.0f}",
                hovertemplate='%{text}<extra></extra>', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x_short[idx]], y=[y_short[idx]], mode='markers',
                marker=dict(color=colors_gradient[i], size=12),
                text=f"Date: {dates[idx]}<br>Traders: {x_short[idx]:,.0f}<br>Position: {-y_short[idx]:,.0f}",
                hovertemplate='%{text}<extra></extra>', showlegend=False
            ))

    # Add regression lines
    if len(x_long) > 1:
        lr_long = LinearRegression().fit(x_long.reshape(-1, 1), y_long)
        x_range_long = np.linspace(x_long.min(), x_long.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range_long, y=lr_long.predict(x_range_long.reshape(-1, 1)),
            mode='lines', name='Long Trend', line=dict(color='#1f77b4', dash='dash'), hoverinfo='skip'
        ))

    if len(x_short) > 1:
        lr_short = LinearRegression().fit(x_short.reshape(-1, 1), y_short)
        x_range_short = np.linspace(x_short.min(), x_short.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range_short, y=lr_short.predict(x_range_short.reshape(-1, 1)),
            mode='lines', name='Short Trend', line=dict(color='#ff7f0e', dash='dash'), hoverinfo='skip'
        ))

    return fig


def create_yearly_color_chart(df_filtered, commodity, subgroup):
    """Create chart with colors by year"""
    df_filtered = df_filtered.copy()
    df_filtered['Year'] = df_filtered['Report_Date'].dt.year
    traders_long_col, traders_short_col, positions_long_col, positions_short_col = get_positioning_columns(subgroup)

    years = sorted(df_filtered['Year'].unique())
    colors = (pc.qualitative.Light24 + pc.qualitative.Dark24)
    year_colors = {year: colors[i % len(colors)] for i, year in enumerate(years)}

    fig = go.Figure()
    legend_entries = set()

    for year in years:
        year_data = df_filtered[df_filtered['Year'] == year]
        show_legend = year not in legend_entries

        fig.add_trace(go.Scatter(
            x=year_data[traders_long_col], y=year_data[positions_long_col],
            mode='markers', name=str(year) if show_legend else '',
            marker=dict(color=year_colors[year], size=9, opacity=0.7, line=dict(width=0.5, color='white')),
            legendgroup=str(year),
            text=[f"Date: {d:%Y-%m-%d}<br>Year: {year}<br>Long Traders: {t:,.0f}<br>Long Position: {p:,.0f}" 
                  for d, t, p in zip(year_data['Report_Date'], year_data[traders_long_col], year_data[positions_long_col])],
            hovertemplate='%{text}<extra></extra>', showlegend=show_legend
        ))
        legend_entries.add(year)

        fig.add_trace(go.Scatter(
            x=year_data[traders_short_col], y=-year_data[positions_short_col],
            mode='markers',
            marker=dict(color=year_colors[year], size=9, opacity=0.7, line=dict(width=0.5, color='white')),
            legendgroup=str(year),
            text=[f"Date: {d:%Y-%m-%d}<br>Year: {year}<br>Short Traders: {t:,.0f}<br>Short Position: {p:,.0f}" 
                  for d, t, p in zip(year_data['Report_Date'], year_data[traders_short_col], year_data[positions_short_col])],
            hovertemplate='%{text}<extra></extra>', showlegend=False
        ))

    # Add regression lines
    x_long_all = df_filtered[traders_long_col].values
    y_long_all = df_filtered[positions_long_col].values
    x_short_all = df_filtered[traders_short_col].values
    y_short_all = -df_filtered[positions_short_col].values

    if len(x_long_all) > 1:
        lr_long = LinearRegression().fit(x_long_all.reshape(-1, 1), y_long_all)
        x_range_long = np.linspace(x_long_all.min(), x_long_all.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range_long, y=lr_long.predict(x_range_long.reshape(-1, 1)),
            mode='lines', name='Long Regression', line=dict(color='black', dash='dash', width=2), hoverinfo='skip'
        ))

    if len(x_short_all) > 1:
        lr_short = LinearRegression().fit(x_short_all.reshape(-1, 1), y_short_all)
        x_range_short = np.linspace(x_short_all.min(), x_short_all.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range_short, y=lr_short.predict(x_range_short.reshape(-1, 1)),
            mode='lines', name='Short Regression', line=dict(color='black', dash='dash', width=2), hoverinfo='skip'
        ))

    return fig


def create_positioning_chart(df_filtered, commodity, subgroup, viz_style):
    """Create positioning chart based on visualization style"""
    if viz_style == 'recent':
        return create_recent_highlight_chart(df_filtered, commodity, subgroup)
    else:  # yearly
        return create_yearly_color_chart(df_filtered, commodity, subgroup)


@app.callback(
    [Output('positioning-chart', 'figure'),
     Output('stats-panel', 'children'),
     Output('date-range', 'children')],
    [Input('commodity-dropdown', 'value'),
     Input('subgroup-dropdown', 'value'),
     Input('viz-style', 'value')]
)
def update_positioning_chart(commodity, subgroup, viz_style):
    """Update the positioning chart based on selections"""
    if not commodity or not subgroup:
        return go.Figure(), [], "Select commodity and trader group"

    df_filtered = df[df['Commodity Name'] == commodity].copy().sort_values('Report_Date')
    fig = create_positioning_chart(df_filtered, commodity, subgroup, viz_style)
    fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.5)

    fig.update_layout(
        title={'text': f'{commodity} Positioning Analysis - {subgroup.replace("_", " ")}', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        xaxis_title=f'Number of {subgroup.replace("_", " ")} Traders',
        yaxis_title='Position Size (Short positions shown as negative)',
        hovermode='closest', template='plotly_white', showlegend=True,
        legend=dict(title="Year" if viz_style == 'yearly' else None, yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1),
        height=650,
        margin=dict(r=150 if viz_style == 'yearly' else 80),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinewidth=1, zerolinecolor='black')
    )

    # Calculate statistics
    traders_long_col, traders_short_col, positions_long_col, positions_short_col = get_positioning_columns(subgroup)

    stats_cards = []
    if not df_filtered.empty:
        current_long_traders = df_filtered[traders_long_col].iloc[-1]
        current_short_traders = df_filtered[traders_short_col].iloc[-1]
        current_long_positions = df_filtered[positions_long_col].iloc[-1]
        current_short_positions = df_filtered[positions_short_col].iloc[-1]
        net_position = current_long_positions - current_short_positions

        stats_cards.extend([
            html.Div([html.H4('Long Traders', style={'color': '#1f77b4'}), html.P(f'{current_long_traders:,.0f}', style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'textAlign': 'center'}),
            html.Div([html.H4('Short Traders', style={'color': '#ff7f0e'}), html.P(f'{current_short_traders:,.0f}', style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'textAlign': 'center'}),
            html.Div([html.H4('Long Positions', style={'color': '#1f77b4'}), html.P(f'{current_long_positions:,.0f}', style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'textAlign': 'center'}),
            html.Div([html.H4('Short Positions', style={'color': '#ff7f0e'}), html.P(f'{current_short_positions:,.0f}', style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'textAlign': 'center'}),
            html.Div([html.H4('Net Position', style={'color': '#2ecc71' if net_position > 0 else '#e74c3c'}), html.P(f'{net_position:+,.0f}', style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'textAlign': 'center'})
        ])

    date_range_text = f"Data Range: {df_filtered['Report_Date'].min():%Y-%m-%d} to {df_filtered['Report_Date'].max():%Y-%m-%d} | Total Reports: {len(df_filtered)}" if not df_filtered.empty else "No data available"

    return fig, stats_cards, date_range_text


@app.callback(
    [Output('concentration-ratio-chart', 'figure'),
     Output('clustering-chart', 'figure'),
     Output('position-size-chart', 'figure')],
    [Input('clustering1-commodity-dropdown', 'value'),
     Input('clustering1-position-dropdown', 'value')]
)
def update_clustering1_charts(commodity, subgroup):
    """Update the three clustering analysis charts"""
    if not commodity or not subgroup:
        return go.Figure(), go.Figure(), go.Figure()

    commodity_df = df[df['Commodity Name'] == commodity].copy()
    subgroup_open_interest_col = f'{subgroup}_All'
    subgroup_traders_col = f'Traders_{subgroup.replace("Positions_", "")}_All'

    required_columns = [subgroup_open_interest_col, 'Open_Interest_All', subgroup_traders_col, 'Traders_Tot_All']
    if not all(col in commodity_df.columns for col in required_columns):
        return go.Figure(), go.Figure(), go.Figure()

    commodity_df['Concentration Ratio'] = np.where(commodity_df['Open_Interest_All'] > 0, commodity_df[subgroup_open_interest_col] / commodity_df['Open_Interest_All'], 0)
    commodity_df['Clustering'] = np.where(commodity_df['Traders_Tot_All'] > 0, commodity_df[subgroup_traders_col] / commodity_df['Traders_Tot_All'], 0)
    commodity_df['Position Size'] = np.where(commodity_df[subgroup_traders_col] > 0, commodity_df[subgroup_open_interest_col] / commodity_df[subgroup_traders_col], 0)

    commodity_df.sort_values('Report_Date', inplace=True)

    fig1 = go.Figure(go.Scatter(x=commodity_df['Report_Date'], y=commodity_df['Concentration Ratio'], mode='lines', name='Concentration Ratio', line=dict(color='#1f77b4', width=2)))
    fig1.update_layout(title=f'{commodity} - {subgroup} Concentration Ratio', xaxis_title='Date', yaxis_title='Ratio', template='plotly_white', height=400)

    fig2 = go.Figure(go.Scatter(x=commodity_df['Report_Date'], y=commodity_df['Clustering'], mode='lines', name='Clustering', line=dict(color='#ff7f0e', width=2)))
    fig2.update_layout(title=f'{commodity} - {subgroup} Clustering', xaxis_title='Date', yaxis_title='Ratio', template='plotly_white', height=400)

    fig3 = go.Figure(go.Scatter(x=commodity_df['Report_Date'], y=commodity_df['Position Size'], mode='lines', name='Position Size', line=dict(color='#2ca02c', width=2)))
    fig3.update_layout(title=f'{commodity} - {subgroup} Position Size', xaxis_title='Date', yaxis_title='Contracts per Trader', template='plotly_white', height=400)

    return fig1, fig2, fig3


@app.callback(
    Output('clustering2-scatter-chart', 'figure'),
    [Input('clustering2-commodity-dropdown', 'value'),
     Input('clustering2-position-dropdown', 'value')]
)
def update_clustering2_chart(commodity, subgroup):
    """Update the clustering scatter plot with gradient coloring"""
    if not commodity or not subgroup:
        return go.Figure()

    commodity_df = df[df['Commodity Name'] == commodity].copy()
    subgroup_open_interest_col = f'{subgroup}_All'
    subgroup_traders_col = f'Traders_{subgroup.replace("Positions_", "")}_All'

    required_columns = [subgroup_open_interest_col, 'Open_Interest_All', subgroup_traders_col, 'Traders_Tot_All']
    if not all(col in commodity_df.columns for col in required_columns):
        return go.Figure()

    commodity_df['Concentration Ratio'] = np.where(commodity_df['Open_Interest_All'] > 0, commodity_df[subgroup_open_interest_col] / commodity_df['Open_Interest_All'], 0)
    commodity_df['Clustering'] = np.where(commodity_df['Traders_Tot_All'] > 0, commodity_df[subgroup_traders_col] / commodity_df['Traders_Tot_All'], 0)

    commodity_df.sort_values('Report_Date', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=commodity_df['Report_Date'], y=commodity_df['Concentration Ratio'], mode='lines', line=dict(color='gray', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=commodity_df['Report_Date'], y=commodity_df['Concentration Ratio'], mode='markers',
        marker=dict(size=8, color=commodity_df['Clustering'], colorscale='RdBu_r', showscale=True, colorbar=dict(title="Clustering %", tickformat='.1%')),
        text=[f"Date: {d:%Y-%m-%d}<br>Concentration: {c:.3f}<br>Clustering: {cl:.1%}" 
              for d, c, cl in zip(commodity_df['Report_Date'], commodity_df['Concentration Ratio'], commodity_df['Clustering'])],
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.update_layout(title=f'{commodity} - {subgroup} Concentration Ratio and Clustering', xaxis_title='Date', yaxis_title='Concentration Ratio', template='plotly_white', height=700, hovermode='closest')

    return fig

@app.callback(
    Output('treemap-chart', 'figure'),
    [Input('week-dropdown', 'value'),
     Input('commodities-filter-dropdown', 'value'),
     Input('treemap-display-mode', 'value')]
)
def update_treemap_chart(week, commodities_filter, display_mode):
    """Simplified and more robust treemap or stacked bar visualization"""
    if not week:
        return go.Figure()

    # Filter weekly data
    df_week = df[df['YYYY Report Week WW'] == week].copy()
    if commodities_filter:
        df_week = df_week[df_week['Commodity Name'].isin(commodities_filter)]

    if df_week.empty:
        return go.Figure(layout={'title': 'No data available for this week'})

    # Define position categories
    position_categories = {
        'Managed Money': 'M_Money_Positions_Long_All',
        'Producer/Merchant': 'Prod_Merc_Positions_Long_All',
        'Swap Dealers': 'Swap_Positions_Long_All',
        'Other Reportables': 'Other_Rept_Positions_Long_All',
        'Non-Reportables': 'NonRept_Positions_Long_All'
    }

    # Prepare data in long format
    df_long = df_week.melt(
        id_vars=['Commodity Name', 'Open_Interest_All'],
        value_vars=list(position_categories.values()),
        var_name='Category_Col',
        value_name='Position'
    )
    df_long['Trader Category'] = df_long['Category_Col'].map({v: k for k, v in position_categories.items()})
    df_long.dropna(subset=['Position'], inplace=True)
    df_long = df_long[df_long['Position'] > 0]

    common_layout = dict(
        title={'text': f'Commodity Positions Breakdown – {week}', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        height=800,
        margin=dict(t=80, l=10, r=10, b=10),
        paper_bgcolor='#f8f9fa',
        template='plotly_white'
    )

    # ---------- Hierarchical Treemap ----------
    if display_mode == 'hierarchical':
        color_map = {
            'Managed Money': '#007bff',
            'Producer/Merchant': '#28a745',
            'Swap Dealers': '#6c757d',
            'Other Reportables': '#ffc107',
            'Non-Reportables': '#17a2b8'
        }

        fig = px.treemap(
            df_long,
            path=['Commodity Name', 'Trader Category'],
            values='Position',
            color='Trader Category',
            color_discrete_map=color_map,
            hover_data={'Position': ':,.0f', 'Open_Interest_All': ':,.0f'}
        )

        fig.update_traces(
            textinfo='label+value+percent parent',
            texttemplate='%{label}<br>%{value:,.0f}<br>%{percentParent:.1%}',
            marker=dict(line=dict(width=2, color='white'))
        )
        fig.update_layout(**common_layout)
        return fig

    # ---------- Stacked Bar ----------
    else:
        color_map = {
            'Managed Money': '#007bff',
            'Producer/Merchant': '#28a745',
            'Swap Dealers': '#6c757d',
            'Other Reportables': '#ffc107',
            'Non-Reportables': '#17a2b8'
        }

        fig = px.bar(
            df_long,
            x='Commodity Name',
            y='Position',
            color='Trader Category',
            color_discrete_map=color_map,
            title=f'Commodity Positions Breakdown – {week}',
            hover_data={'Position': ':,.0f'}
        )

        fig.update_layout(
            barmode='stack',
            hovermode='x unified',
            **common_layout
        )
        return fig


@app.callback(
    [Output('regression-chart', 'figure'),
     Output('regression-summary-table', 'children')],
    [Input('regression-commodity-dropdown', 'value'),
     Input('regression-category-dropdown', 'value')]
)
def update_regression_analysis(commodity, category):
    """Update the regression plot and correlation table"""
    if not commodity or not category or price_df.empty:
        return go.Figure(layout={'title': 'Please select a commodity and category'}), []

    cot_comm_df = df[df['Commodity Name'] == commodity].copy()
    price_comm_df = price_df[price_df['Commodity Name'] == commodity].copy()

    if cot_comm_df.empty or price_comm_df.empty:
        return go.Figure(layout={'title': f'No price or COT data available for {commodity}'}), []

    merged_df = pd.merge(cot_comm_df, price_comm_df, left_on='Report_Date', right_on='Date', how='inner').sort_values('Report_Date').reset_index(drop=True)

    if len(merged_df) < 2:
        return go.Figure(layout={'title': 'Not enough overlapping data to calculate changes'}), []

    merged_df['Price_Change'] = merged_df['Price'].diff()
    categories = {'M_Money': 'Managed Money', 'Prod_Merc': 'Producer/Merchant', 'Swap': 'Swap Dealers', 'Other_Rept': 'Other Reportables', 'NonRept': 'Non-Reportables'}
    corr_data = []

    for code, label in categories.items():
        long_col, short_col = (f'Swap_Positions_Long_All', f'Swap_Positions_Short_All') if code == 'Swap' else (f'{code}_Positions_Long_All', f'{code}_Positions_Short_All')
        if long_col in merged_df.columns and short_col in merged_df.columns:
            merged_df[f'{code}_Net'] = merged_df[long_col] - merged_df[short_col]
            merged_df[f'{code}_Net_Change'] = merged_df[f'{code}_Net'].diff()
            correlation = merged_df['Price_Change'].corr(merged_df[f'{code}_Net_Change'])
            corr_data.append({'Category': label, 'Correlation': correlation})

    merged_df.dropna(subset=['Price_Change', f'{category}_Net_Change'], inplace=True)
    if merged_df.empty:
        return go.Figure(layout={'title': 'No valid data points after calculating weekly changes.'}), []

    y_col = f'{category}_Net_Change'
    fig = px.scatter(merged_df, x='Price_Change', y=y_col, trendline='ols', trendline_color_override='red',
                     labels={'Price_Change': 'Weekly Price Change', y_col: f'Weekly Net Position Change ({categories[category]})'},
                     hover_data={'Report_Date': '|%Y-%m-%d'})
    fig.update_layout(title=f'{commodity}: Price Change vs. {categories[category]} Net Position Change', template='plotly_white')

    corr_df = pd.DataFrame(corr_data).dropna().sort_values('Correlation', ascending=False)
    header = [html.Thead(html.Tr([html.Th('Trader Category'), html.Th('Correlation')]))]
    body = [html.Tbody([html.Tr([html.Td(row['Category']), html.Td(f"{row['Correlation']:.3f}", style={'color': 'green' if row['Correlation'] > 0 else 'red'})]) for _, row in corr_df.iterrows()])]
    summary_table = html.Table(header + body, className='table')

    return fig, summary_table


# Expose the server for deployment
server = app.server

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)


# In[ ]:




