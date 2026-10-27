"""Builds the interactive map: a heatmap driven by the clusters, on real map tiles."""
# ^ docstring: a short note about this file. python does not run it, it is only for humans.
#
# big picture: this file does NOT draw anything on screen by itself. it builds a "folium Map" object in python.
# folium is a python wrapper around Leaflet, a JavaScript map library. our python code describes the map
# (layers, heat points, markers) and folium turns that description into HTML + JavaScript.
# later, ui/result_view.py hands the map to the browser (through st_folium), and the browser runs the JavaScript.

# lets us write type hints like "str | None" without errors on older python versions.
from __future__ import annotations

# html has a tool called escape() that makes text safe to put inside a web page (see the popups below).
import html

# folium builds the map. numpy is for fast maths on arrays. pandas is for tables (DataFrames).
import folium
import numpy as np
import pandas as pd
# Fullscreen = a small button that makes the map fill the screen. HeatMap = the glowing heat layer.
from folium.plugins import Fullscreen, HeatMap
# ConvexHull finds the outline around a group of points (like a rubber band stretched around them).
from scipy.spatial import ConvexHull

# summarise_clusters (from summary.py) gives one row per cluster: size, centre, top offences.
# the dot in ".summary" means "the file summary.py in this same folder (core)".
from .summary import summarise_clusters

# a list of 16 colours (hex codes). cluster 1 gets the first colour, cluster 2 the second, and so on.
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b",
    "#bcbd22", "#393b79", "#ad494a", "#637939", "#7b4173", "#e7ba52", "#3182bd", "#843c39",
]
# grey colour for noise points (crimes that belong to no cluster).
NOISE_COLOUR = "#8a8f98"

# the three ways to colour the heat. the dictionary maps the text the user sees (left)
# to a short code the program uses (right). the map settings radio buttons are built from this.
HEAT_MODES = {
    "Cluster size (bigger hotspot = hotter)": "cluster_size",
    "Equal (shows where the clusters are)": "equal",
    "Plain density (ignores the clustering)": "density",
}


# gives the colour for a cluster number.
def cluster_colour(cluster_index: int) -> str:
    # "% len(PALETTE)" is the remainder after dividing by 16. it makes the colours repeat after 16 clusters,
    # so the 17th cluster gets the first colour again. int() makes sure we have a whole number.
    return PALETTE[int(cluster_index) % len(PALETTE)]


# works out how "hot" each crime should be (a number from 0 to 1). the leading _ means "helper, used only in this file".
# a small example: labels = [0, 0, 0, 1, -1] and mode "cluster_size"
# gives [1.0, 1.0, 1.0, 0.58, 0.0]. cluster 0 has 3 crimes, cluster 1 has 1 crime, the last crime is noise.
def _heat_weights(labels: np.ndarray, mode: str, include_noise: bool) -> np.ndarray:
    # True for crimes that are in a cluster, False for noise (-1).
    clustered = labels >= 0
    # "plain density" ignores the clusters: every crime counts the same (1), noise included.
    if mode == "density":
        return np.ones(len(labels))
    # otherwise start with 0 (cold) for every crime. 0 means "do not draw this crime in the heat layer".
    weights = np.zeros(len(labels))
    # only do this if there is at least one clustered crime.
    if clustered.any():
        # bigger clusters should glow hotter.
        if mode == "cluster_size":
            # np.bincount counts how many crimes each cluster number has. sizes[0] = size of cluster 0, and so on.
            sizes = np.bincount(labels[clustered])
            # square root keeps small clusters visible next to a huge one
            # for each clustered crime: look up its cluster's size, divide by the biggest size (gives 0 to 1),
            # then take the square root. without the root, a tiny cluster next to a huge one would be almost invisible.
            weights[clustered] = np.sqrt(sizes[labels[clustered]] / sizes.max())
            # force the weights to stay between 0.2 and 1.0, so even the smallest cluster still shows a little.
            weights[clustered] = np.clip(weights[clustered], 0.2, 1.0)
        # mode "equal": every clustered crime gets the full weight 1.
        else:
            weights[clustered] = 1.0
    # if the user asked for it, give noise crimes a faint weight (0.1) instead of 0.
    if include_noise:
        weights[~clustered] = 0.1
    # give back one weight per crime.
    return weights


# this is the main function of the file. it builds and returns the whole map.
# df = the table of crimes (lat, lon, offence). labels = the cluster number of each crime.
# the "*" means: everything after it must be given by name, like heat_mode="equal" (not by position).
# the values after "=" are the defaults used when the caller does not give them.
def build_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    # which heat style to use (see HEAT_MODES above).
    heat_mode: str = "cluster_size",
    # radius = how big each heat point glows (in screen pixels). blur = how soft the glow edges are.
    radius: int = 15,
    blur: int = 12,
    # the faintest the heat is allowed to be. 0 = fully see-through, 1 = solid.
    min_opacity: float = 0.3,
    # show noise crimes faintly in the heat?
    include_noise: bool = False,
    # draw a circle at the centre of each cluster?
    show_centres: bool = True,
    # draw an outline around each cluster?
    show_hulls: bool = False,
    # draw a small dot for crimes?
    show_points: bool = False,
    # draw at most this many dots (lots of dots make the page heavy and slow).
    max_points: int = 1500,
    # draw at most this many clusters (the biggest ones).
    max_clusters_drawn: int = 100,
    # "-> folium.Map" means the function gives back a folium Map object.
) -> folium.Map:
    # pull the latitude and longitude columns out of the table as plain number arrays.
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    # make sure labels is a numpy array.
    labels = np.asarray(labels)
    # n = the number of crimes.
    n = len(lat)

    # create the empty map. location = the first centre (the middle value of lat and lon).
    # zoom_start = how zoomed in it starts. tiles=None = no default background, because we add our own just below.
    # control_scale=True adds a small distance scale bar. prefer_canvas=True draws shapes on a canvas,
    # which is faster when there are many dots.
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=12, tiles=None, control_scale=True, prefer_canvas=True)
    # the first background: the OpenStreetMap street map. name = the label in the layer menu. show=True = visible at the start.
    folium.TileLayer("OpenStreetMap", name="Street map", show=True).add_to(m)
    # a map is made of many small square pictures called tiles. the browser asks a tile server for the tiles
    # of the part you are looking at. in the web address, {z} = zoom level, {x} and {y} = which tile.
    # this one is a light grey map from Esri. max_zoom = the most you can zoom in on it. show=False = hidden until chosen.
    # attr = the credit text shown in the corner (providers ask for it).
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light grey map", max_zoom=16, show=False,
    ).add_to(m)
    # a satellite picture background, also from Esri. also hidden until chosen.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", max_zoom=19, show=False,
    ).add_to(m)

    # zoom to where most of the data is, ignoring a few far-away stragglers
    # np.percentile(lat, [1, 99]) gives the value that 1% of the points are below, and the value that 99% are below.
    # using these instead of the smallest and biggest value means a few far-away crimes cannot zoom the map out too far.
    south, north = np.percentile(lat, [1, 99])
    west, east = np.percentile(lon, [1, 99])
    # if all the crimes are at almost the same place (less than 0.001 degrees, about 100 m, apart),
    # add a small margin so the map does not zoom in as far as it can go.
    if north - south < 1e-3:
        south, north = south - 0.005, north + 0.005
    if east - west < 1e-3:
        west, east = west - 0.005, east + 0.005
    # move and zoom the map so this box fits on screen. padding adds 20 pixels of space around it.
    m.fit_bounds([[float(south), float(west)], [float(north), float(east)]], padding=(20, 20))

    # heatmap
    # one heat number (0 to 1) for every crime.
    weights = _heat_weights(labels, heat_mode, include_noise)
    # only crimes with a weight above 0 are drawn. this also keeps the page smaller.
    visible = weights > 0
    # if there is at least one crime to draw.
    if visible.any():
        # the heat layer wants a list of [latitude, longitude, weight]. column_stack glues the three arrays side by side.
        # .round(5) cuts each number to 5 decimals (about 1 metre), which makes the page smaller.
        # .tolist() changes the numpy array into a normal python list.
        heat_data = np.column_stack([lat[visible], lon[visible], weights[visible]]).round(5).tolist()
        # create the heat layer and put it on the map. name = its label in the layer menu.
        # int(...) and float(...) make sure the numbers have the right type.
        HeatMap(heat_data, name="Heatmap", radius=int(radius), blur=int(blur), min_opacity=float(min_opacity)).add_to(m)

    # one row per cluster, biggest first (cluster numbers here start at 1).
    summary = summarise_clusters(df, labels)
    # keep only the biggest clusters, so the page does not get too heavy.
    drawn = summary.head(max_clusters_drawn)

    # outlines around the clusters. "and len(drawn)" means: only if there is at least one cluster to draw.
    if show_hulls and len(drawn):
        # a FeatureGroup is a folder of shapes that can be switched on and off together from the layer menu.
        group = folium.FeatureGroup(name="Cluster outlines", show=True)
        # walk through the clusters one by one. each "row" is one cluster, with names like row.cluster and row.incidents.
        for row in drawn.itertuples():
            # find the positions of the crimes in this cluster. the summary starts counting at 1
            # but the labels start at 0, so we subtract 1. flatnonzero gives the positions where the test is True.
            idx = np.flatnonzero(labels == row.cluster - 1)
            # an outline needs at least 3 points. skip tiny clusters.
            if len(idx) < 3:
                continue
            # a table of [latitude, longitude] for the crimes in this cluster.
            pts = np.column_stack([lat[idx], lon[idx]])
            # try to find the outline. this can fail, for example when all points are on one straight line.
            try:
                # ConvexHull(pts).vertices = the positions of the corner points of the outline.
                # pts[...] picks those corner points, and .tolist() makes it a normal list.
                outline = pts[ConvexHull(pts).vertices].tolist()
            except Exception:  # all points on one line, etc.
                # if it failed, skip this cluster and carry on with the next one.
                continue
            # this cluster's colour (again subtracting 1 to match the colour list).
            colour = cluster_colour(row.cluster - 1)
            # draw the outline as a shape. fill=True with fill_opacity=0.10 gives it a very light colour inside.
            # tooltip = the text shown when you hover over it. :, adds commas to big numbers, like 1,200.
            folium.Polygon(
                outline, color=colour, weight=2, fill=True, fill_opacity=0.10,
                tooltip=f"Cluster {row.cluster}: {row.incidents:,} incidents",
            ).add_to(group)
        # put the whole folder of outlines on the map.
        group.add_to(m)

    # small dots for individual crimes.
    if show_points:
        # another switchable folder, for the dots.
        group = folium.FeatureGroup(name="Incident dots", show=True)
        # a random number generator. the 0 is a fixed seed, so we pick the same dots every time.
        rng = np.random.default_rng(0)
        # choose at most max_points crimes at random (replace=False means no crime is picked twice).
        for i in rng.choice(n, size=min(n, int(max_points)), replace=False):
            # grey for noise, otherwise the colour of the crime's cluster.
            colour = NOISE_COLOUR if labels[i] < 0 else cluster_colour(labels[i])
            # the words shown when you hover: "noise" or "cluster 3" (+ 1 because we count clusters from 1).
            where = "noise" if labels[i] < 0 else f"cluster {labels[i] + 1}"
            # draw one small circle. weight=0 means no border line. fill_opacity 0.75 = a bit see-through.
            # html.escape makes the offence name safe for a web page: characters like < and & are changed,
            # so a strange offence name in the file cannot break the page.
            folium.CircleMarker(
                [lat[i], lon[i]], radius=3, color=colour, weight=0, fill=True, fill_color=colour, fill_opacity=0.75,
                tooltip=f"{html.escape(str(df['offence'].iloc[i]))} ({where})",
            ).add_to(group)
        # put the folder of dots on the map.
        group.add_to(m)

    # a circle at the centre of each cluster, with a click-popup.
    if show_centres and len(drawn):
        # another switchable folder, for the centre circles.
        group = folium.FeatureGroup(name="Cluster centres", show=True)
        # the size of the biggest cluster. we use it to scale the circles.
        biggest = float(drawn["incidents"].max())
        # walk through the clusters one by one.
        for row in drawn.itertuples():
            # this cluster's colour.
            colour = cluster_colour(row.cluster - 1)
            # the text that appears when you click the circle. <b> = bold, <br> = new line (this is HTML).
            # :.1f shows one decimal. the offence names are escaped for safety, like above.
            popup = (
                f"<b>Cluster {row.cluster}</b><br>{row.incidents:,} incidents ({row.share_pct:.1f}% of all)<br>"
                f"Top offences: {html.escape(str(row.top_offences))}"
            )
            # draw the circle at the cluster's centre (the average latitude and longitude).
            # radius: bigger clusters get bigger circles, from 6 up to 18 pixels (square root keeps it reasonable).
            # white border, filled with the cluster colour. tooltip = hover text. popup = click text, at most 300 px wide.
            folium.CircleMarker(
                [row.centre_lat, row.centre_lon],
                radius=6 + 12 * float(np.sqrt(row.incidents / biggest)),
                color="#ffffff", weight=2, fill=True, fill_color=colour, fill_opacity=0.9,
                tooltip=f"Cluster {row.cluster}: {row.incidents:,} incidents",
                popup=folium.Popup(popup, max_width=300),
            ).add_to(group)
        # put the folder of centre circles on the map.
        group.add_to(m)

    # add a button that makes the map fill the whole screen.
    Fullscreen().add_to(m)
    # add the layer menu (top right). it lets the user switch the backgrounds, heatmap, outlines, dots and centres on and off.
    # collapsed=True keeps it as a small icon until you hover over it.
    folium.LayerControl(collapsed=True).add_to(m)
    # give the finished map back. result_view.py will show it on the page.
    return m
