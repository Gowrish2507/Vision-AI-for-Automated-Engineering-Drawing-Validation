import plotly.graph_objects as go

def calc_deviation(detected, estimated):
    if detected == 0: return 0
    return round(abs(detected - estimated) / detected * 100, 2)

def get_smart_recommendation(component, detected, estimated, tolerance):
    if detected == 0 and estimated == 0: return None
    dev = abs(detected - estimated) / detected * 100 if detected > 0 else 100
    if dev > tolerance:
        return f"⚠️ **{component}**: You estimated {estimated}, Drawing suggests ~{detected:.1f} (Deviation: {dev:.1f}%)"
    return None

def create_comparison_graph(est_area, est_wall, det_area, det_wall):
    """Generates the Estimated vs Detected Bar Chart."""
    fig = go.Figure(data=[
        go.Bar(name='Estimated', x=['Area (m²)', 'Wall Length (m)'], y=[est_area, est_wall], marker_color='royalblue'),
        go.Bar(name='Detected', x=['Area (m²)', 'Wall Length (m)'], y=[det_area, det_wall], marker_color='darkorange')
    ])
    fig.update_layout(barmode='group', title='Estimated vs Detected Values', yaxis_title='Value')
    return fig

def create_material_graph(room_names, room_tiles, room_paints):
    """Generates the Room-wise Material Bar Chart."""
    fig = go.Figure(data=[
        go.Bar(name='Tiles (pcs)', x=room_names, y=room_tiles, marker_color='brown'),
        go.Bar(name='Paint (Liters)', x=room_names, y=room_paints, marker_color='teal'),
    ])
    fig.update_layout(barmode='group', title='Room-Wise Material Requirements', yaxis_title='Quantity')
    return fig