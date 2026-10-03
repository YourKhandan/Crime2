# Crime hotspot clustering

A Streamlit app that finds crime hotspots in a file of offence locations, draws them as a heatmap
on a real map, and scores K-Means, DBSCAN, KNN and Infomap so you can see which fits your data best.

## Run it

```bash
cd crime_hotspot_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at http://localhost:8501. The map needs an internet connection for its tiles.

## Your file

A CSV or .xlsx with a latitude, a longitude and an offence column (any names; the app guesses and
lets you change them). Every other column is ignored. Rows with invalid coordinates are reported and,
by default, left out.

No data yet? Tick "Use the built-in sample data". It is synthetic, made up for testing.

## Folders

```
app.py                  page and tabs
algorithms/             one file per algorithm (kmeans_algo, dbscan_algo, knn_algo, infomap_algo)
core/validation.py      latitude/longitude checks
core/metrics.py         Silhouette, Davies-Bouldin, Dunn
core/heatmap.py         the map
ui/                     the three tabs
make_sample_data.py     regenerates sample_crime_data.csv
```

## Adding an algorithm

Copy any file in `algorithms/`, change NAME, DESCRIPTION, PARAMS and `run()`, then add it to
`algorithms/__init__.py`. Its parameter widgets, scores and heatmap appear automatically.

## Good to know

- Distances are in metres (points are projected from latitude/longitude first), so eps and distance limits mean real metres.
- Plain KNN needs labelled data, so the KNN option groups points using their nearest-neighbour lists (Jarvis-Patrick style).
- Silhouette, Davies-Bouldin and Dunn ignore noise points and favour round clusters, so read them next to the heatmaps.
- Infomap is the slowest: about 15 seconds for 50,000 points. The others take about a second.
- On big data, Silhouette and Dunn use a sample. Data above "Maximum points to cluster" is sampled at random.
