import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any, List
import json

class DataVisualizationService:
    def __init__(self):
        pass

    def create_line_chart(self, data: Dict[str, List[float]], title: str, x_label: str, y_label: str) -> str:
        fig = go.Figure()
        for key, values in data.items():
            fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode='lines+markers', name=key))
        
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return pio.to_json(fig)

    def create_bar_chart(self, data: Dict[str, float], title: str, x_label: str, y_label: str) -> str:
        fig = go.Figure(data=[go.Bar(x=list(data.keys()), y=list(data.values()))])
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return pio.to_json(fig)

    def create_pie_chart(self, data: Dict[str, float], title: str) -> str:
        fig = go.Figure(data=[go.Pie(labels=list(data.keys()), values=list(data.values()))])
        fig.update_layout(title=title)
        return pio.to_json(fig)

data_visualization_service = DataVisualizationService()

def get_data_visualization_service() -> DataVisualizationService:
    return data_visualization_service
