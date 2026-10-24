"""Crime hotspot clustering lab.

Run with:  python streamlit -m run app.py
"""
import streamlit as st

from ui import cluster_tab, compare_tab, data_tab
# Streamlit is an open-source Python framework used to visualize data science projects... 
st.set_page_config(page_title="Crime hotspot clustering", page_icon="🗺️", layout="wide")#browser tab title 

st.title("Crime hotspot clustering")
# written on the first website on web
st.write(
    "Upload offence locations, check the coordinates, then find hotspots with K-Means, DBSCAN, Louvain or Infomap "
    "and compare them on a heatmap and three quality scores."
)
# as streamlit handles spacing on its own... so makes 3 tabs 
tab_data, tab_cluster, tab_compare = st.tabs(["Data and validation", "Cluster and heatmap", "Compare algorithms"])
with tab_data:
    dataset = data_tab.render()#for first tab we go into data_tab inside ui folder

with tab_cluster:
    if dataset is None:
        st.info("Load and validate your data in the first tab, then come back here.")
    else:
        cluster_tab.render(dataset)

with tab_compare:
    if dataset is None:
        st.info("Load and validate your data in the first tab, then come back here.")
    else:
        compare_tab.render(dataset)
