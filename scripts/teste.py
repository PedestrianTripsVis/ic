import contextily as cx
import matplotlib.pyplot as plt
import geopandas as gpd

db = gpd.read_file("./maps/Se-Distrito_modified/Se-Distrito_modified.shp")

w, s, e, n = db.to_crs(epsg=3857).total_bounds
img, ext = cx.bounds2raster(w, s, e, n, source=cx.providers.CartoDB.Positron)