# ShadeLA Optimization Project
shadeLA is a spatial data project focused on analyzing and visualizing the relationship between shade coverage, heat vulnerability, and environmental equity across Los Angeles County.

It integrates multiple geospatial datasets — including vegetation, built environment, demographic, and climate data — to identify areas most in need of cooling interventions such as tree planting or shade infrastructure.

shadeLA/
│
├── data/
│   ├── bus_stops.geojson
│   ├── ECOSTRESS_LST.tif
│   ├── DTLA_Venue_Buffer.geojson
│   ├── la_excess_er.geojson
│   ├── la28_venues.geojson
│   ├── metro_stations.geojson
│   ├── metrolink_stations.geojson
|   ├── shade.geojson
|   ├── social_sensitivity.geojson
|   ├── tree_canopy.geojson
|   ├── vacant_tree_park.geojson
|   └── vacant_tree_street.geojson
|
├── notebooks/
│   └── 01_data_cleaning_visualizations.ipynb
|
│
└── README.md



## 🗺️ Data Layers and Sources

| Layer | Description | Source | Format |
|--------|--------------|----------|----------|
| **Bus Stops/ Metro Stops / Metrolink Stations** | Public Transportation Stations/Stops. | [Los Angeles Metro GIS Data Portal](https://developer.metro.net/gis-data/) | Point |
| **Land Surface Temperature (C)** | Land Surface Temperature (LST) for streets in LA | [NASA JPL ECOSTRESS](https://ecostress.jpl.nasa.gov/data) | Raster (10mx10m)|
| **DTLA Venue Buffer** | Zone around DTLA that will be converted to a pedestrian only zone during the game. | LA Department of Transporation (private data) | Polygon |
| **Los Angeles Excess ER Visits** | The rate per 10,000 people for excess visits to the ER on extremely hot day. | [UCLA Heat Maps](https://uclaheatmaps.org/) | Polygon (Zipcode) |
| **Shade Data** | Data for shade and tree canopy coverage at different times of day| [UCLA Luskin & American Forests](https://www.treeequityscore.org/methodology?tab=data-download) | Polygon (Census Tract)  |
| **Social Sensitivity** | Los Angeles County Analysis of social and environmental risk to heat. | [LA County](https://lacounty.maps.arcgis.com/apps/webappviewer/index.html?id=c78e929d004846bb993958b49c8e8e65) | Polygon (Census Tract)|
| **Vacant Trees** | Locations of vacant tree wells in parks and along streets. | [City of Los Angeles](https://losangelesca.treekeepersoftware.com/index.cfm?deviceWidth=2560) | Point |

