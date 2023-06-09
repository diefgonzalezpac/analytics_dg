# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, exceptions
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc
import dash_daq as daq
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Incorporate data
# power_bi_data = pd.read_excel('Power_bi_data\Clean_Total_power_BI_data.xlsx')
#power_bi_data = pd.read_csv('Power_bi_data\Total_insights_page_data.csv')
# power_bi_data = pd.read_csv('Power_bi_data\Total_insights_page_data_nonan.csv')
power_bi_data = pd.read_parquet('Power_bi_data\Total_insights_page_data_nonan.parquet',engine='fastparquet')
power_bi_data["Engine_cycles_general_percentage"] = power_bi_data["Total Pages"]*0.0002
power_bi_data["Engine_cycles_model_percentage"] = (power_bi_data["Total Pages"] * 100) / power_bi_data["Limit"]
power_bi_data["Engine_cycles_model_percentage"] = power_bi_data["Engine_cycles_model_percentage"].replace([np.nan,np.inf,-np.inf], 0)
# power_bi_data = power_bi_data[power_bi_data["Geo Market"]!="North America (NA)"]


external_df = pd.DataFrame()
number_of_bins_init = 200
number_of_bins = 80


""" This function receives a dataframe ,a column name and group size and returns a list with the relation between
the amount of errors and the number of printers (serialnumber) in each group """

def group_of_errors_sum(df:pd.DataFrame,data_series:pd.Series,color_column:str ,group_size:int): 
    bins = np.linspace(data_series.min(), data_series.max(), group_size+1)
    df['Bins'] = pd.cut(data_series, bins=bins)
    df["midpoint"] = df["Bins"].apply(lambda x: round(x.mid,2))
    # color_column_count = df[df[color_column] != 0].shape[0] 
    count_nonzero = lambda x: (x != 0).sum()
    grouped_data = df.groupby('Bins').agg(sum_color_column = (color_column,'sum'),Serial_Number = ('Serial Number','count'),new_color_column=(color_column,count_nonzero))
    # print(grouped_data)
    grouped_data['norm_error_cost'] = grouped_data["sum_color_column"]/grouped_data['Serial_Number']
    #grouped_data['norm_error_cost'] = grouped_data[sum_color_column]/color_column_count
    #grouped_data['norm_error_cost'] = grouped_data["sum_color_column"]/grouped_data['new_color_column']
    grouped_data['norm_error_cost'] = grouped_data['norm_error_cost'].fillna(0)
    grouped_data['norm_error_cost'] = grouped_data['norm_error_cost'].apply(lambda x: round(x,2))
    # print(grouped_data)
    # print(grouped_data['norm_error_cost'].sum())
    # print(grouped_data['norm_error_cost'].max())
    grouped_data = grouped_data.reset_index()
    grouped_data["midpoint"] = grouped_data["Bins"].apply(lambda x: round(x.mid,2))
    return grouped_data

def dynamic_axis_limit(max_value):
    if max_value > 300:
        return 300
    else:
        return math.ceil(max_value.item())
    
def calculate_next_hundred(max_value):
    next_hundred = ((max_value // 100) + 1) * 100
    return next_hundred

def pages_axis_range(limit,number_bins_x):
    return (limit * number_bins_x)/2

def histogram_fig(df_model:pd.DataFrame,column_name:str,number_of_bins:int):
    grouped_data = group_of_errors_sum(df_model,df_model[column_name],'Incidents',number_of_bins)
    color_midpoint = grouped_data["norm_error_cost"].quantile(0.75)
    #print(grouped_data)
    # fig = px.bar(x=grouped_data["midpoint"], y=grouped_data["Serial_Number"],color=grouped_data["norm_error_cost"],\
    #          color_continuous_scale=[(0.00,"#008000"), (0.1,"#008000"),
    #                                  (0.1,"#94ff00"), (0.2,"#94ff00"),
    #                                  (0.2,"#ECFF00"), (0.4,"#ECFF00"),
    #                                  (0.4,"#FFFF00"), (0.6,"#FFFF00"),
    #                                  (0.6,"#F70000"), (0.8,"#F70000"),
    #                                  (0.8,"#8B0000"), (1,"#8B0000")],\
    #          labels={'x':'Engine Cycles Percentage','y':'Number of printers','color':'Average incidents'})
    fig = px.bar(x=grouped_data["midpoint"], y=grouped_data["Serial_Number"],color=grouped_data["norm_error_cost"],\
             color_continuous_scale='Portland',range_color=(0,color_midpoint),\
             labels={'x':'Engine Cycles Percentage','y':'Number of printers','color':'Average incidents'})
    fig.update_xaxes(title_text='Engine Cycles Percentage %')
    #fig.update_xaxes(range=[0, 2_000_000])
    limit = dynamic_axis_limit(df_model[column_name].max())
    fig.update_xaxes(range=[0, limit])
    fig.update_yaxes(title_text='Number of printers')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'white'},\
                      autosize=True,bargap=0)
    
    return fig

def correlation_fig(df_model:pd.DataFrame,my_flag:bool):
    if my_flag:
        fig_c = px.scatter(df_model, x="Engine_cycles_model_percentage", y="Incidents", color="Model", trendline="ols",
                           trendline_color_override="#ef553b",color_discrete_sequence=['#f4d44d'],hover_data=["Serial Number"])
  
        # cost_array = df_model["total_cost_per_incident"].values
        # cost_per_incident_array = fig_c.data[1].y * cost_array

        df = pd.DataFrame({
            "total_cost":df_model["total_cost_per_incident"]*(fig_c.data[1].y),
            "cost_per_incident":df_model["total_cost_per_incident"]
        })

        customdata = ['<BR><b>Average cost * incidents: </b>' + str(element1) + 
              '<BR><b>Cost per incident: </b>' + str(element2) 
              for element1, element2 in zip(df["total_cost"], df["cost_per_incident"])]
        

        # fig_c.data[1].update(customdata=cost_per_incident_array, 
        #                      hovertemplate='<b>OLS trendline</b><br>Incidents = 0.00278884 * Engine_cycles_model_percentage + 0.395058<br>R<sup>2</sup>=0.024970<br><br>Model=HP LaserJet Enterprise M506dn Printer<br>Engine_cycles_model_percentage=%{x}<br>Incidents=%{y} <b>(trend)</b><br>Acumulative Reparation Cost:%{customdata:.2f} (USD)<extra></extra>')
        # fig_c.data[1].update(customdata=cost_per_incident_array, 
        #                      hovertemplate=fig_c.data[1].hovertemplate + \
        #                      '<br>Reparation Cost:%{customdata[{0}]:.2f} (USD)<extra></extra>')

        fig_c.data[1].update(customdata=customdata, 
                             hovertemplate=fig_c.data[1].hovertemplate + \
                             '%{customdata}')

        fig_c.add_trace(go.Scatter(x=df_model["Total Pages"], y=np.nan*(df_model["Incidents"]),
                                   xaxis='x2', name='Second X Axis',showlegend=False))
        
    
        model_limit = int(df_model["Limit"].iloc[0])
        percentage_x_range = calculate_next_hundred(df_model["Engine_cycles_model_percentage"].max())
        print(percentage_x_range)
        number_of_bins_x = int(percentage_x_range/50)
        pages_axis_range_value = pages_axis_range(model_limit,number_of_bins_x)

        fig_c.update_layout(xaxis=dict(autorange=False,position=0,tickvals=np.linspace(0,percentage_x_range,number_of_bins_x+1), range=[0,percentage_x_range]),
                            xaxis2=dict(position=0.9,tickvals=np.linspace(0,pages_axis_range_value,number_of_bins_x+1), 
                                        range=[0,pages_axis_range_value], side='top',overlaying='x',
                                        anchor='y',showgrid=False,autorange=False))   

    else:
        fig_c = px.scatter(df_model, x="Engine_cycles_general_percentage", y="Incidents", color="Model", trendline="ols",
                 trendline_scope="overall",hover_data=["Serial Number"])
        
    fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'white'},\
                        autosize=True,xaxis=dict(showgrid=False),yaxis=dict(showgrid=True))
    

    return fig_c

def pie_fig(df_model:pd.DataFrame):
    total = sum(df_model['count'])
    fig_pie = px.pie(df_model, values='count', names='text_count')
    fig_pie.update_layout(autosize=True,margin=dict(l=20, r=20, t=20, b=20),\
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'white'})
    fig_pie.update_traces(textposition='inside',hovertemplate='<b>%{label}</b><br>Devices: %{value}<br>Percentage:%{percent}')
    fig_pie.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    return fig_pie



# Initialize the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# app = Dash(__name__)

modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(id='click_info'),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col([
                            html.P("Total Devices in Group"),
                            daq.LEDDisplay(
                                id="operator-led",
                                value=0,
                                color="#92e0d3",
                                backgroundColor="#1e2130",
                                size=50,
                            )
                        ],width=4),
                        dbc.Col([
                            html.P("Total Incidents in Group"),
                            daq.LEDDisplay(
                                id="operator-led2",
                                value=0,
                                color="#92e0d3",
                                backgroundColor="#1e2130",
                                size=50,
                            ),
                        ],width=4),
                        dbc.Col([
                            html.P("Rate of incidents"),
                            daq.LEDDisplay(
                                id="operator-led3",
                                value='',
                                color="#92e0d3",
                                backgroundColor="#1e2130",
                                size=50,
                            ),
                        ],width=4)

                    ])
                ]),
            ],
            id="modal",
            centered=True
            # size="lg",
        ),
    ]
)

app.layout = dbc.Container([
    html.Div(children=[
        html.H2('Print Services'),
        html.Img(id="logo_hp",src=app.get_asset_url('hp_logo.png'),alt="Logo", style={'width':'8%','height':'8%','margin-left':'10px'}), #,style={'width':'10%','height':'10%'}
    ],style={'display': 'flex'}),
    dbc.Row([
        dcc.Dropdown(id="mydropdown",
                     options = power_bi_data['Model'].unique(),
                     value = None),
    ]),
    dbc.Row([
        dbc.Col(
                dcc.Graph(
                    id='my_graph',
                    figure={}
                ),
        align='start',
        width=8
        ),
        dbc.Col(
                id='container',
        align='end',
        width=4
        )
    ],justify='center'),
    modal,
    dbc.Row([
        dcc.Graph(
                    id='correlation_graph',
                    figure={}
                )
    ]),
])


# @app.callback(
#     Output("modal", "is_open"),
#     Output("hover_info","children"),
#     [Input("my_graph", "hoverData"), Input("close", "n_clicks")],
#     [State("modal", "is_open")],
# )
# def toggle_modal(hover_data,close_button, is_open):
#     if hover_data or close_button:
#         x = hover_data['points'][0]['x']
#         y = hover_data['points'][0]['y']
#         text = "x = "+str(x)+" & y = "+str(y)
#         return not is_open,text
#     return is_open,None

@app.callback(
    Output("my_graph", "figure"),
    [Input("mydropdown", "value")]
)

def fig_graph(printer_selector):
    external_df = power_bi_data
    if printer_selector == None:
        external_df = power_bi_data
        fig = histogram_fig(external_df,'Engine_cycles_general_percentage',number_of_bins_init)
    else:
        external_df = power_bi_data[power_bi_data['Model']==printer_selector]
        fig = histogram_fig(external_df,'Engine_cycles_model_percentage',number_of_bins)
    return fig#,external_df.to_json(orient='split')


@app.callback(
    Output("modal", "is_open"),
    Output("operator-led","value"),
    Output("operator-led2","value"),
    Output("operator-led3","value"),
    Output("my_graph", "clickData"),
    [Input("my_graph", "clickData")],
    [State("modal", "is_open"),State("mydropdown", "value")],
)
def fig_click(clickData,is_open,printer_selector):
    if not clickData:
        raise exceptions.PreventUpdate
        
    if clickData:
        # external_df = pd.read_json(data_store, orient='split')
        # print(external_df.columns)
        if printer_selector == None:
            external_df = power_bi_data
            group_of_errors_sum(external_df,external_df['Engine_cycles_general_percentage'],'Incidents',number_of_bins_init)
        else:
            external_df = power_bi_data[power_bi_data['Model']==printer_selector]
            group_of_errors_sum(external_df,external_df['Engine_cycles_model_percentage'],'Incidents',number_of_bins)

        if clickData:
            info = round(clickData["points"][0]["x"],2)
            number_of_printers = clickData["points"][0]["y"]
            #number_of_printers_text = "Number of printer in this group: " + str(number_of_printers) 
            bin_data = external_df[external_df["midpoint"]==info]
            total_incidents = int(bin_data['Incidents'].sum())
            devices_with_incidents = len(bin_data[bin_data['Incidents']!=0.0]) * 100
            devices_with_incidents_percent = str(round(devices_with_incidents/number_of_printers,2))
            return not is_open,number_of_printers,total_incidents,devices_with_incidents_percent, None
    return is_open, None,None

@app.callback(
    Output("correlation_graph", "figure"),
    [Input("mydropdown", "value"),Input("correlation_graph", "clickData")]
)

def fig_graph(printer_selector,click_data):
    if printer_selector == None:
        fig_c = correlation_fig(power_bi_data,my_flag=False)
    else:
        fig_c = correlation_fig(power_bi_data[power_bi_data['Model']==printer_selector],my_flag=True)
        
    if click_data and click_data.get('event',{}).get('dblclick'):
        fig_c.update_xaxes(autorange=False)

    return fig_c


@app.callback(
    Output("container", "children"),
    [Input("my_graph", "hoverData"),Input("mydropdown", "value")]
)

def pie_graph_generator(hover_data,printer_selector):
    if hover_data is None:
         return None
    else:
        if printer_selector == None:
            external_df = power_bi_data
            group_of_errors_sum(external_df,external_df['Engine_cycles_general_percentage'],'Incidents',number_of_bins_init)
        else:
            external_df = power_bi_data[power_bi_data['Model']==printer_selector]
            group_of_errors_sum(external_df,external_df['Engine_cycles_model_percentage'],'Incidents',number_of_bins)

        x_position = hover_data['points'][0]['x']
        bin_data = external_df[external_df["midpoint"]==x_position]
        bin_data["text_count"] = bin_data["Incidents"].astype(str) + " incidents"
        bin_data = bin_data["text_count"].value_counts().reset_index().sort_values(by="text_count")
        pie_graph = pie_fig(bin_data)
        
        return dcc.Graph(figure= pie_graph)





# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)



# fig_cm = make_subplots(shared_xaxes=True,shared_yaxes=True)
# fig_c = px.scatter(df_model, x="Engine_cycles_model_percentage", y="Incidents", color="Model", trendline="ols",
#                     trendline_color_override="#ef553b",color_discrete_sequence=['#f4d44d'],hover_data=["Serial Number"])

# cost_array = df_model["total_cost_per_incident"].values
# cost_per_incident_array = fig_c.data[1].y * cost_array
# # fig_c.data[1].update(customdata=cost_per_incident_array, 
# #                      hovertemplate='<b>OLS trendline</b><br>Incidents = 0.00278884 * Engine_cycles_model_percentage + 0.395058<br>R<sup>2</sup>=0.024970<br><br>Model=HP LaserJet Enterprise M506dn Printer<br>Engine_cycles_model_percentage=%{x}<br>Incidents=%{y} <b>(trend)</b><br>Acumulative Reparation Cost:%{customdata:.2f} (USD)<extra></extra>')
# fig_c.data[1].update(customdata=cost_per_incident_array, 
#                         hovertemplate=fig_c.data[1].hovertemplate + '<br>Reparation Cost:%{customdata:.2f} (USD)<extra></extra>')
# fig_c.add_trace(go.Scatter(x=df_model["Engine_cycles_model_percentage"], y=np.nan*(df_model["Incidents"]), 
#                         xaxis='x2', name='Second X Axis',showlegend=False))
# fig_c.update_layout(
#     xaxis2=dict(
#         anchor='y',
#         title='Second X Axis',
#         overlaying='x',
#         side='top',
#         tickmode='sync'
#     )
# )
