import pandas as pd
import plotly.express as px

from data_preprocess import preprocess_data  # Assuming preprocess_data is defined elsewhere

def visualize_data():
    data = preprocess_data()

    # Bar Graphs of Features: Rating, Effectiveness, & Side Effects
    fig_1 = px.histogram(data, x='rating')
    fig_1.update_layout(xaxis_title='Rating', yaxis_title='Number of Reviews')
    fig_1.update_xaxes(showgrid=False)
    fig_1.update_yaxes(showgrid=False)
    fig_1.show()
    fig_1.write_image('fig_1.jpg')

    effectiveness_order = ['Highly Effective', 'Considerably Effective', 'Moderately Effective', 'Marginally Effective', 'Ineffective']
    side_effects_order = ['No Side Effects', 'Mild Side Effects', 'Moderate Side Effects', 'Severe Side Effects', 'Extremely Severe Side Effects']

    fig_2 = px.histogram(data, x='effectiveness', category_orders={'effectiveness': effectiveness_order})
    fig_2.update_layout(xaxis_title='Effectiveness', yaxis_title='Number of Reviews')
    fig_2.update_xaxes(showgrid=False)
    fig_2.update_yaxes(showgrid=False)
    fig_2.show()
    fig_2.write_image('fig_2.jpg')

    fig_3 = px.histogram(data, x='sideEffects', category_orders={'sideEffects': side_effects_order})
    fig_3.update_layout(xaxis_title='Side Effects', yaxis_title='Number of Reviews')
    fig_3.update_xaxes(showgrid=False)
    fig_3.update_yaxes(showgrid=False)
    fig_3.show()
    fig_3.write_image('fig_3.jpg')

    # Stacked Graph for Effectiveness & Side Effects
    color_palette = ['#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08306B'] # Define a custom darker color palette

    fig_4 = px.histogram(data, x='effectiveness', color='sideEffects',
                         category_orders={'effectiveness': effectiveness_order, 'sideEffects': side_effects_order},
                         color_discrete_sequence=color_palette)
    
    fig_4.update_layout(xaxis_title='Effectiveness', yaxis_title='Number of Reviews', legend_title='Side Effects')
    fig_4.update_xaxes(showgrid=False)
    fig_4.update_yaxes(showgrid=False)
    fig_4.show()
    fig_4.write_image('fig_4.jpg')

    return data

visualize_data()
