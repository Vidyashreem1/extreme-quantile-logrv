# Rainfall Input Data

The rainfall data used in the manuscript are not redistributed in this
repository. To run the conditional analysis, arrange the inputs as follows:

```text
data/
|-- western_ghats_coordinates.csv
`-- Western ghats data/
    |-- data_<latitude>_<longitude>.csv
    `-- ...
```

The coordinate file must contain the columns `lat`, `lon`, and `alt_mean`.
Each grid-level rainfall file is a headerless CSV file named
`data_<latitude>_<longitude>.csv`. It must contain seven columns in the
following order: `Year`, `Month`, `Day`, `Precipitation`, `MaxTemp`,
`MinTemp`, and `MeanTemp`.

Alternative locations can be supplied through `CENN_LDNN_COORD_FILE` and
`CENN_LDNN_RAIN_DIR`.
