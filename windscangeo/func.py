import os
import re

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import s3fs
import xarray as xr
from sklearn.neighbors import BallTree
from tqdm import tqdm

def vectorized_solar_angles(lat, lon, time_utc):

    """
    This function calculates the solar zenith angle (SZA) and solar azimuth angle (SAA) for a given latitude, longitude, and time. This is an archived function. Current implementation does not use solar angles but only image input.

    Args:
        lat (numpy.ndarray): The latitude values of the scatterometer data.
        lon (numpy.ndarray): The longitude values of the scatterometer data.
        time_utc (numpy.ndarray): The observation times in UTC.

    Returns:
        sza (numpy.ndarray): The solar zenith angle in degrees.
        saa (numpy.ndarray): The solar azimuth angle in degrees.
    """

    # Convert time to Julian Day
    timestamp = pd.to_datetime(time_utc).tz_localize(None)
    jd = (
        timestamp.astype("datetime64[ns]").astype(np.int64) / 86400000000000 + 2440587.5
    )
    d = jd - 2451545.0  # Days since J2000

    # Mean longitude, mean anomaly, ecliptic longitude
    g = np.deg2rad((357.529 + 0.98560028 * d) % 360)  # Mean anomaly
    q = np.deg2rad((280.459 + 0.98564736 * d) % 360)  # Mean longitude
    L = (q + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)) % (
        2 * np.pi
    )  # Ecliptic long

    # Obliquity of the ecliptic
    e = np.deg2rad(23.439 - 0.00000036 * d)

    # Sun declination
    sin_delta = np.sin(e) * np.sin(L)
    delta = np.arcsin(sin_delta)

    # Equation of time (in minutes)
    E = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )

    # Convert time to fractional hours (UTC)
    fractional_hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600

    # Solar time correction
    time_offset = E + 4 * lon  # lon in degrees
    tst = fractional_hour * 60 + time_offset  # True Solar Time in minutes
    ha = np.deg2rad((tst / 4 - 180) % 360)  # Hour angle in radians

    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat)

    # Solar zenith angle
    cos_zenith = np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(
        delta
    ) * np.cos(ha)
    zenith = np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))  # in degrees

    # Solar saa angle
    sin_saa = -np.sin(ha) * np.cos(delta)
    cos_saa = np.cos(lat_rad) * np.sin(delta) - np.sin(lat_rad) * np.cos(
        delta
    ) * np.cos(ha)
    saa = np.rad2deg(np.arctan2(sin_saa, cos_saa))
    saa = (saa + 360) % 360  # Normalize

    return zenith, saa

def create_folder(experiment_name):
    """
    Create a folder for saving results based on the experiment name.

    Args:
        experiment_name (str): Name of the experiment to create a folder for.
        If the folder already exists, it will not be created again.

    Returns:
        str: Path to the created folder.
    """

    path_folder = f"./results_folder/model_day_{experiment_name}"

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
        print(f"Folder created at {path_folder}")

    return path_folder



def get_indices(lat_grid, lon_grid, Goeslat, Goeslon, radius=0.125):
    """
    Finds the corresponding GOES row and column indices for each scatterometer point
    using a BallTree for efficiency, and then filtering points to form a square bounding box.

    Args:
        lat_grid (numpy.ndarray): 2D array of latitudes from the scatterometer data.
        lon_grid (numpy.ndarray): 2D array of longitudes from the scatterometer data.
        Goeslat (numpy.ndarray): 2D array of latitudes from the GOES data.
        Goeslon (numpy.ndarray): 2D array of longitudes from the GOES data.
        radius (float): Radius in degrees to define the bounding box around each scatterometer point.
    Returns:
        indices_array (numpy.ndarray): 2D array of tuples, where each tuple contains the row and column indices of the corresponding GOES pixel for each scatterometer point.
        

    """

    print("INFO : Calculating indices")
    # Flatten GOES data
    Goeslat_flat = Goeslat.flatten()
    Goeslon_flat = Goeslon.flatten()
    goes_points = np.column_stack((Goeslat_flat, Goeslon_flat))

    # Build BallTree with haversine distance
    goes_points_rad = np.radians(goes_points)
    goes_tree = BallTree(goes_points_rad, metric="haversine")

    # Flatten scatter grids
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    scatter_points = np.column_stack((lat_flat, lon_flat))
    scatter_points_rad = np.radians(scatter_points)

    # Radius for broad-phase query: diagonal of the bounding box
    # Square box ±radius: diagonal = radius * sqrt(2)
    diag_radius = radius * np.sqrt(2)
    diag_radius_rad = np.radians(diag_radius)

    indices_array = np.empty(lat_flat.shape, dtype=object)
    goes_shape = Goeslat.shape

    for i, (lat_val, lon_val) in enumerate(zip(lat_flat, lon_flat)):
        # Broad-phase: query all points within diagonal distance
        candidate_indices = goes_tree.query_radius(
            np.array([scatter_points_rad[i]]), r=diag_radius_rad
        )[0]

        if candidate_indices.size == 0:
            # No points found, store empty
            indices_array[i] = (np.array([], dtype=int), np.array([], dtype=int))
            continue

        # Post-filter candidates to keep only those in the bounding box
        lat_min = lat_val - radius
        lat_max = lat_val + radius
        lon_min = lon_val - radius
        lon_max = lon_val + radius

        cand_lats = Goeslat_flat[candidate_indices]
        cand_lons = Goeslon_flat[candidate_indices]

        mask = (
            (cand_lats >= lat_min)
            & (cand_lats <= lat_max)
            & (cand_lons >= lon_min)
            & (cand_lons <= lon_max)
        )

        final_indices = candidate_indices[mask]

        # Convert these flat indices back to row,col
        rows, cols = np.unravel_index(final_indices, goes_shape)
        indices_array[i] = (rows, cols)

    # Reshape indices_array to the original shape
    indices_array = indices_array.reshape(lat_grid.shape)
    return indices_array



def calculate_degrees(file_id):
    """This function calculates the latitude and longitude of the GOES ABI fixed grid projection. 
    This function comes from NOAA/NESDIS/STAR. (2025). Latitude and longitude remapping of GOES-R ABI imagery using Python . Atmospheric Composition Science Team. Retrieved from https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php

    Args:
        file_id (xarray.Dataset): The xarray dataset containing the GOES ABI fixed grid projection variables.

    Returns:
        abi_lat (numpy.ndarray): The latitude of the GOES ABI fixed grid projection.
        abi_lon (numpy.ndarray): The longitude of the GOES ABI fixed grid projection.
    

    """

    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables["y"][:]  # N/S elevation angle in radians
    projection_info = file_id.goes_imager_projection
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    abi_lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))
        )
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    print("INFO : Latitude and longitude calculated")
    return abi_lat, abi_lon


def index_parallel(ds, ScatterDataset):
    """
    Finds the corresponding GOES row and column indices for the entire scatterometer dataset.

    Args:
        ScatterDataset: xarray Dataset containing scatterometer data.
        scatter_name: Name for the output file.
        output_path: Path to save the output file.

    Returns:
        parallel_indice_values: 2D array of tuples containing GOES row and column indices corresponding to scatterometer data.
    """

    create_folder("satellite_indices")
    ds_spatial_resolution = ds.spatial_resolution
    ds_spatial_resolution.replace(" ", "_")

    name_str = f"lat_{ScatterDataset.latitude.min().values}_{ScatterDataset.latitude.max().values}_lon_{ScatterDataset.longitude.min().values}_{ScatterDataset.longitude.max().values}_res_{ds_spatial_resolution}"
    name_str = name_str.replace(".", "_")
    if os.path.exists(
        f"./satellite_indices/{ds_spatial_resolution}_index.npy"
    ):
        parallel_index = np.load(
            f"./satellite_indices/{ds_spatial_resolution}_index.npy",
            allow_pickle=True,
        )

        return parallel_index

    else:
        print(
            "INFO : Satellite index file not found, creating new index file. This might take a while."
        )

        # Extract scatterometer latitudes and longitudes
        Latitudes_Scatter = ScatterDataset["latitude"].values
        Longitudes_Scatter = ScatterDataset["longitude"].values

        # Create a meshgrid of scatterometer coordinates
        lon_grid, lat_grid = np.meshgrid(Longitudes_Scatter, Latitudes_Scatter)

        # Extract GOES latitudes and longitudes
        Goeslat, Goeslon = calculate_degrees(ds)
        Goeslat[np.isnan(Goeslat)] = 999
        Goeslon[np.isnan(Goeslon)] = 999
        # Use the optimized get_indices function
        parallel_indice_values = get_indices(lat_grid, lon_grid, Goeslat, Goeslon)

        # Save the indices array
        np.save(
            f"./satellite_indices/{ds_spatial_resolution}_index.npy",
            parallel_indice_values,
        )

        return parallel_indice_values


def sort_by_time(lat_list, lon_list, time_list, wind_speed_list):
    """
    This function sorts the output of savedataseperated() by time.
    This allows for more efficient data processing and allows file caching for times that are represented by the same GOES file.

    Args:
        lat_list (numpy.ndarray): The latitude values of the scatterometer data.
        lon_list (numpy.ndarray): The longitude values of the scatterometer data.
        time_list (numpy.ndarray): The measurement time values of the scatterometer data.
        wind_speed_list (numpy.ndarray): The wind speed values of the scatterometer data.

    Returns:
        lat_list_sorted (numpy.ndarray): The sorted latitude values of the scatterometer data.
        lon_list_sorted (numpy.ndarray): The sorted longitude values of the scatterometer data.
        time_list_sorted (numpy.ndarray): The sorted measurement time values of the scatterometer data.
        wind_speed_list_sorted (numpy.ndarray): The sorted wind speed values of the scatterometer data.

    """
    # Get the indices that would sort the measurement_time array
    sorted_indices = np.argsort(time_list)

    # Reorder the arrays using the sorted indices
    time_list_sorted = time_list[sorted_indices]
    lat_list_sorted = lat_list[sorted_indices]
    lon_list_sorted = lon_list[sorted_indices]
    speed_list_sorted = wind_speed_list[sorted_indices]

    return lat_list_sorted, lon_list_sorted, time_list_sorted, speed_list_sorted


def savedataseperated(ScatterData, main_parameter,verbose=True):
    """
    This function extracts the valid lon / lat / measurement time and the main parameter from ever pixel
    of the scatterometer data and saves it to a numpy file.

    Args:
        ScatterData (xarray.Dataset): The ASCAT dataset containing the scatterometer data.
        main_parameter (xarray.DataArray): The main parameter to be saved. This can be a classification / wind speed / wind direction etc.

    Returns:

        lat_list (numpy.ndarray): The latitude values of the scatterometer data.
        lon_list (numpy.ndarray): The longitude values of the scatterometer data.
        time_list (numpy.ndarray): The measurement time values of the scatterometer data.
        main_parameter_list (numpy.ndarray): The main parameter values of the scatterometer data.

    this function saves the data locally to a folder called data_processed_scat
    """
    lat_full, lon_full, time_full = ScatterData.indexes.values()
    measurement_time_full = ScatterData.measurement_time

    lat_full = np.array(lat_full)
    lon_full = np.array(lon_full)
    measurement_time_full = np.array(measurement_time_full)
    main_parameter = np.array(main_parameter)

    index = np.argwhere(~np.isnan(main_parameter))

    index_list = []
    lat_list = []
    lon_list = []
    time_list = []
    wind_speed_list = []

    name_scatter = ScatterData.source

    for t, i, j in index:

        # print(t,'= time', i,'=row', j, '=column')
        index_list.append((t, i, j))

        # print(measurement_time_full[t, i, j].astype('datetime64[ns]'))
        time_list.append(measurement_time_full[t, i, j])

        # print(lat_full[i])
        lat_list.append(lat_full[i])

        # print(lon_full[j])
        lon_list.append(lon_full[j])

        # print(AllWindSpeeds[t, i, j])
        wind_speed_list.append(main_parameter[t, i, j])

    lat_list = np.array(lat_list)
    lon_list = np.array(lon_list)
    time_list = np.array(time_list)
    wind_speed_list = np.array(wind_speed_list)

    lat_list, lon_list, time_list, wind_speed_list = sort_by_time(
        lat_list, lon_list, time_list, wind_speed_list
    )
    if verbose:
        save_overpass_time(time_list,name_scatter)    

    return lat_list, lon_list, time_list, wind_speed_list

def save_overpass_time(time_list,name_scatter):
    """
    This function prints the overpass time of the scatterometer.

    Args:
        time_list (numpy.ndarray): The measurement time values of the scatterometer data.
        name_scatter (str): The name of the scatterometer data source (e.g. ASCAT, HYSCAT etc).

    Returns:
        None 

    """
    formated_time = time_list.astype('datetime64[ns]')
    hour_minute = formated_time.astype('datetime64[m]')
    unique_hour_minute = np.unique(hour_minute)

    filtered = [unique_hour_minute[0]]

    delta = np.timedelta64(1, 'h')

    for time in unique_hour_minute[1:]:
        if time - filtered[-1] >= delta:
            filtered.append(time)

    time_only = []
    for time in filtered:
        time = str(time).split('T')[1]
        time_only.append(time)
    print(f"ORBIT : {name_scatter} overpass time : {time_only}")

def get_goes_url(time, goes_aws_url_folder='noaa-goes16/ABI-L2-CMIPF', goes_channel="C01"):
    """
    This function gets the nearest GOES-16 files from the time given.
    The function returns a list of urls to the files.
    The function uses the s3fs library to access the AWS GOES-16 data.

    Args:
        time (numpy.datetime[ns]): The time of the scatterometer data.
        goes_aws_url_folder (str): The folder in the AWS S3 bucket where the GOES-16 data is stored.
        goes_channel (list): The channel of interest.

    Returns:
        urls (list): A list of urls to the GOES-16 files.


    """
    date_c = time.astype("datetime64[ns]")
    date = pd.to_datetime(date_c)
    date_str = date.strftime("%Y/%j/%H")
    min = int(date.strftime("%M"))
    min_range = [(min + i) % 60 for i in range(-6, 7)]
    min_range_str = [f"{x:02d}" for x in min_range]
    fs = s3fs.S3FileSystem(anon=True)
    # get the nearest goes file from time

    urls = []
    channel = goes_channel
    path = f"{goes_aws_url_folder}/{date_str}"
    files = fs.ls(path)
    filter_channel = [x for x in files if channel in x]
    if len(filter_channel) == 0:
        print(f"INFO :No file found for {channel} on day {date_str}, skipping file")
        return
    file = [x for x in filter_channel if x[73:75] in min_range_str]
    if len(file) == 0:
        print(
            f"INFO :No file found for {channel} on day {date_str} for minute {min}, skipping file"
        )
        return np.zeros(len(goes_channel))
    urls.append("s3://" + file[0])

    return urls


def extract_scatter(
    polar_data,
    date,
    lat_range,
    lon_range,
    verbose=True,
    main_variable="wind_speed",
):
    """
    This function extracts the scatterometer data from the polar_data dataset for the given time range, latitude range and longitude range.
    The function then saves the data into 4 numpy files : time of observation, latitude, longitude and main variable.

    Args:
        polar_data (xarray.Dataset): The scatterometer dataset (ASCAT, HYSCAT etc).
        date (numpy.datetime64): The time of the scatterometer data.
        lat_range (tuple): The latitude range of the scatterometer data.
        lon_range (tuple): The longitude range of the scatterometer data.
        verbose (bool): If True, the function will print the progress of the extraction.
        main_variable (str): The main variable to be extracted from the scatterometer data. This can be wind speed, wind direction, classification etc.

    Returns:
        observation_times (numpy.ndarray): The time of observation of the scatterometer data.
        observation_lats (numpy.ndarray): The latitude of the scatterometer data.
        observation_lons (numpy.ndarray): The longitude of the scatterometer data.
        observation_main_parameter (numpy.ndarray): main parameter extracted (wind_speed).

    """

    polar = polar_data.sel(
        time=slice(date, date),
        latitude=slice(lat_range[0], lat_range[1]),
        longitude=slice(lon_range[0], lon_range[1]),
    )

    seperated_scatter = savedataseperated(polar, polar[main_variable],verbose=verbose)

    observation_times = seperated_scatter[2]
    observation_lats = seperated_scatter[0]
    observation_lons = seperated_scatter[1]
    observation_wind_speeds = seperated_scatter[3]

    

    return (
        observation_times,
        observation_lats,
        observation_lons,
        observation_wind_speeds,
    )


def get_image(ds, parallel_index, lat_grd, lon_grd, lat_search, lon_search,goes_image_size=128):

    """
    This function retrieves a trainable GOES image for a given latitude and longitude from a GOES16 `.nc` file.

    Args:
        ds (xarray.Dataset): The xarray dataset containing the GOES data.
        parallel_index (numpy.ndarray): The precomputed indices for GOES pixels corresponding to scatterometer lat/lon.
        lat_grd (numpy.ndarray): The latitude grid of the scatterometer data.
        lon_grd (numpy.ndarray): The longitude grid of the scatterometer data.
        lat_search (float): The latitude to search for in the GOES data.
        lon_search (float): The longitude to search for in the GOES data.
        goes_image_size (int): The size of the output image. Default is 128.

    Returns:
        padded_image (xarray.DataArray): A padded xarray DataArray containing the GOES image centered around the specified lat/lon.

    """
    index_row = np.where(
        lat_grd == lat_search,
    )
    index_column = np.where(lon_grd == lon_search)

    rows_goes = parallel_index[index_row[0][0], index_column[0][0]][0]
    columns_goes = parallel_index[index_row[0][0], index_column[0][0]][1]

    if rows_goes.size == 0 or columns_goes.size == 0:
        return None

    pixels_from_center = (goes_image_size-1) // 2
    mean_row = rows_goes.mean().astype(int)
    min_row = mean_row - pixels_from_center
    max_row = mean_row + pixels_from_center

    mean_col = columns_goes.mean().astype(int)
    min_col = mean_col - pixels_from_center
    max_col = mean_col + pixels_from_center

    if "CMI" in ds: # If using GOES-16 L2 processed data
        image = ds.CMI[min_row:max_row, min_col:max_col].values

    elif "Rad" in ds: #If using GOES-16 L1b data
        image = ds.Rad[min_row:max_row, min_col:max_col].values

    # debug
    # print(min_row,'= min_row', max_row,'= max_row', min_col, '= min_col', max_col, '= max_col')
    target_size = (goes_image_size, goes_image_size)

    padded_image = np.pad(
        image,
        (
            (
                (target_size[0] - image.shape[0]) // 2,
                (target_size[0] - image.shape[0] + 1) // 2,
            ),
            (
                (target_size[1] - image.shape[1]) // 2,
                (target_size[1] - image.shape[1] + 1) // 2,
            ),
        ),
        constant_values=0,
    )

    padded_image = xr.DataArray(padded_image, dims=("x", "y"))
    return padded_image

def extract_goes(
    observation_times,
    observation_lats,
    observation_lons,
    scatterometer_data_path,
    goes_aws_url_folder,
    goes_channel="C01",
    goes_image_size=128,
    verbose=True,
):
    """
    This function extracts GOES images for the given observation times, latitudes, and longitudes.
    It retrieves the GOES data from the specified AWS S3 bucket and processes it to create images of the specified size.

    Args:
        observation_times (numpy.ndarray): The times of observation of the scatterometer data. 
        observation_lats (numpy.ndarray): The latitudes of the scatterometer data.
        observation_lons (numpy.ndarray): The longitudes of the scatterometer data.
        scatterometer_data_path (str): The path to the scatterometer data directory.
        goes_aws_url_folder (str): The folder in the AWS S3 bucket where the GOES data is stored.
        goes_channel (str): The channel of interest. Default is "C01".
        goes_image_size (int): The size of the output images. Default is 128.
        verbose (bool): If True, prints progress information.
    
    Returns:
        images (numpy.ndarray): A 4D numpy array of shape (num_observations, num_channels, goes_image_size, goes_image_size) containing the extracted GOES images.

    """

    for file in os.listdir(scatterometer_data_path):
        if file.endswith(".nc"):
            polar = xr.open_dataset(
                os.path.join(scatterometer_data_path, file),
                engine="h5netcdf",
                drop_variables=["DQF"],
            )
            break
        
        else:
            print('WARNING : No .nc file found in the scatterometer data path, please check the path')

    template_scatter = polar.isel(time=0)
    lat_grd, lon_grd = (
        template_scatter["latitude"].values,
        template_scatter["longitude"].values,
    )

    fs = fsspec.filesystem("s3", anon=True, default_block_size=512 * 1024**1024)

    values, counts = np.unique(observation_times, return_counts=True)

    all_urls = []  # getting unique URLS
    for value in values:
        urls = get_goes_url(value, goes_aws_url_folder,goes_channel)
        all_urls.append(urls)

    values_url, indices_url, counts_url = np.unique(
        all_urls, return_index=True, return_counts=True, axis=0
    )
    # Sort indices to "unsort" the URLs
    sorted_indices = sorted(range(len(indices_url)), key=lambda k: indices_url[k])
    values_url = [all_urls[indices_url[i]] for i in sorted_indices]

    # Reorder counts_url using the same sorted indices
    counts_url = [counts_url[i] for i in sorted_indices]

    compressed_urls = values_url
    compressed_counts = []
    start_idx = 0

    for size in counts_url:
        group_sum = counts[start_idx : start_idx + size].sum()
        compressed_counts.append(group_sum)
        start_idx += size

    width = goes_image_size
    height = goes_image_size

    images = np.zeros([len(observation_times), 1 , width, height], dtype=np.float32)

    total_idx = 0
    for unique_idx, unique_urls in tqdm(
        enumerate(compressed_urls),
        desc="INFO : Retrieving and processing GOES data",
        total=len(compressed_urls),
        disable=not verbose,
    ):

    
        for CH_idx, url_CH in enumerate(unique_urls):

            if url_CH == 0:
                images[total_idx, CH_idx] = np.zeros([width, height])
                continue

            with fs.open(url_CH, mode="rb") as f:

                ds = xr.open_dataset(
                    f, engine="h5netcdf", drop_variables=["DQF"]
                )  # this is the bottleneck

                parallel_index = index_parallel(
                    ds,
                    template_scatter,
                )
                for i in range(compressed_counts[unique_idx]):
                    images[total_idx + i, CH_idx] = get_image(
                        ds=ds,
                        parallel_index=parallel_index,
                        lat_grd=lat_grd,
                        lon_grd=lon_grd,
                        lat_search=observation_lats[total_idx + i],
                        lon_search=observation_lons[total_idx + i],
                        goes_image_size=goes_image_size,
                    )

        total_idx += compressed_counts[unique_idx]

    if verbose:
        print(
            f"INFO : Extracted {len(observation_times)} images from {len(compressed_urls)} GOES files."
        )
    return images




def goes_index(parallel_index, lat_grd, lon_grd, lat_search, lon_search):
    """
    This function retrieves the indices of the GOES image corresponding to a given latitude and longitude. This is an archived function. Current implementation decides on extent based on chosen image size.

    Args:
        parallel_index (numpy.ndarray): The precomputed indices for GOES pixels corresponding to scatterometer lat/lon.
        lat_grd (numpy.ndarray): The latitude grid of the scatterometer data.
        lon_grd (numpy.ndarray): The longitude grid of the scatterometer data.
        lat_search (float): The latitude to search for in the GOES data.
        lon_search (float): The longitude to search for in the GOES data.

    Returns:
        min_row (int): The minimum row index of the GOES image.
        max_row (int): The maximum row index of the GOES image.
        min_col (int): The minimum column index of the GOES image.
        max_col (int): The maximum column index of the GOES image.
    """

    index_row = np.where(lat_grd == lat_search)
    index_column = np.where(lon_grd == lon_search)

    rows_goes = parallel_index[index_row[0][0], index_column[0][0]][0]
    columns_goes = parallel_index[index_row[0][0], index_column[0][0]][1]

    if rows_goes.size == 0 or columns_goes.size == 0:
        return None

    min_row = rows_goes.min()
    max_row = rows_goes.max()

    min_col = columns_goes.min()
    max_col = columns_goes.max()

    return min_row, max_row, min_col, max_col


def filter_invalid(
    images,
    numerical_data,
    min_nonzero_pixels=50,
):
    
    """
    This function filters out invalid images and corresponding numerical data based on two criteria:
    1) The sum of pixel values in the image is not zero (i.e., the image is not completely empty).
    2) The number of non-zero pixels in the image is greater than or equal to a specified minimum threshold (default is 50).

    Args:
        images (numpy.ndarray): A 4D numpy array of shape (num_images, num_channels, height, width) containing the GOES images.
        numerical_data (dict): A dictionary containing numerical data associated with the images. The keys should match the dimensions of the images.
        min_nonzero_pixels (int): The minimum number of non-zero pixels required for an image to be considered valid. Default is 50.

    Returns:
        filtered_images (numpy.ndarray): A 4D numpy array of shape (num_valid_images, num_channels, height, width) containing the filtered GOES images.
        filtered_numerical_data (dict): A dictionary containing the numerical data associated with the valid images.

    """
    # Sums of pixel values in each image
    sums_images = [np.nansum(x) for x in images]

    # Counts of non-zero pixels in each image
    nonzero_counts = [np.count_nonzero(x) for x in images]

    # Build a "mask_invalid" array of indices that fail any criterion:
    # 1) sum == 0 (completely empty)
    # 2) nonzero pixel count < min_nonzero_pixels (not enough data)

    mask_valid = np.where(
        (np.array(sums_images) != 0) & (np.array(nonzero_counts) >= min_nonzero_pixels)
    )[0]

    # Delete the invalid entries from each array
    filtered_numerical_data = {
        key: value[mask_valid] for key, value in numerical_data.items()
    }
    filtered_images = images[mask_valid]
    n_removed_images = len(images) - len(filtered_images)

    print(
        "INFO : Filtered invalid images. Removed {} entries.".format(
            n_removed_images
        )
    )
    return (
        filtered_images,
        filtered_numerical_data,
    )


def fill_nans(images):
    """
    This function fills NaN values in the images with zeros. (This is simply np.nan_to_num)

    Args:
        images (numpy.ndarray): A 4D numpy array of shape (num_images, num_channels, height, width) containing the GOES images.
    
    Returns:
        images (numpy.ndarray): A 4D numpy array with NaN values replaced by zeros.
    """
    images = np.nan_to_num(images, nan=0.0)
    print("INFO : Filled nans")
    return images


def filter_nighttime(
    observation_times,
    observation_lats,
    observation_lons,
    observation_wind_speeds,
    min_hour=10,
    max_hour=19,
    verbose=True,
):
    """
    This function filters the scatterometer data to only include observations that were made during daylight hours.
    The function checks the hour of each observation time and only keeps those that fall within the specified
    range (default is 10 to 19, which corresponds to 10 AM to 7 PM UTC).

    Args:
        observation_times (numpy.ndarray): The times of observation of the scatterometer data.
        observation_lats (numpy.ndarray): The latitudes of the scatterometer data.
        observation_lons (numpy.ndarray): The longitudes of the scatterometer data.
        observation_wind_speeds (numpy.ndarray): The wind speeds of the scatterometer data.
        min_hour (int): The minimum hour of the day to include (default is 10).
        max_hour (int): The maximum hour of the day to include (default is 19).
        verbose (bool): If True, prints the number of valid scatterometer data points at daylight.
    
    Returns:
        valid_times (list): A list of valid observation times that fall within the specified hour range.
        valid_lats (list): A list of valid latitudes corresponding to the valid observation times
        valid_lons (list): A list of valid longitudes corresponding to the valid observation times.
        valid_wind_speeds (list): A list of valid wind speeds corresponding to the valid observation times.

    """

    valid_times = []
    valid_lats = []
    valid_lons = []
    valid_wind_speeds = []

    for idx in range(len(observation_times)):
        only_hour = int(
            observation_times[idx].astype("datetime64[ns]").astype("str")[11:13]
        )
        if min_hour <= only_hour <= max_hour:
            valid_times.append(observation_times[idx])
            valid_lats.append(observation_lats[idx])
            valid_lons.append(observation_lons[idx])
            valid_wind_speeds.append(observation_wind_speeds[idx])

    if verbose:
        print(f"INFO : Total number of scatterometer data points at daylight : {len(valid_times)}")
    return valid_times, valid_lats, valid_lons, valid_wind_speeds




def package_data(
    images,
    numerical_data,
    filter=True,
    solar_conversion=False,
    verbose=True
):
    """
    This function packages the images and numerical data into a format that can be used for training a machine learning model.
    The function will filter out invalid images and fill in any NaN values. (Invalid images = empty images from GOES data)
    The function will also convert the observation times, latitudes and longitudes to solar angles (sza, saa) if solar_conversion is set to True.
    The function will return the images and numerical data in a numpy array format.

    Args:
        images (numpy.ndarray): The GOES images corresponding to the observation data.
        numerical_data (dict): A dictionary containing the numerical data corresponding to the observation data. The keys should include "observation_lats", "observation_lons", "observation_times" and optionally "wind_speeds".
        filter (bool): If True, the function will filter out invalid images and fill in Nan values. Default is True.
        solar_conversion (bool): If True, the function will convert the observation times, latitudes and longitudes to solar angles (sza, saa). Default is False. (Not used in current implementation, but kept in case of future use)
        verbose (bool): If True, the function will print progress information. Default is True.

    Returns:
        images (numpy.ndarray): The GOES images corresponding to the observation data.
        numerical_data (numpy.ndarray): The numerical data corresponding to the observation data. (sza, saa, main_parameter if solar_conversion is set to True or lat, lon, time, wind_speeds if solar_conversion is set to False)

    """
    if filter:
        (images, numerical_data) = filter_invalid(images, numerical_data)
        images = fill_nans(images)

    if solar_conversion:
        observation_lats = numerical_data["observation_lats"]
        observation_lons = numerical_data["observation_lons"]
        observation_times = numerical_data["observation_times"]

        sza, saa = vectorized_solar_angles(
            observation_lats, observation_lons, observation_times
        )

        sza_rad = np.deg2rad(sza)
        sza_sin = np.sin(sza_rad)
        sza_cos = np.cos(sza_rad)

        saa_rad = np.deg2rad(saa)
        saa_sin = np.sin(saa_rad)
        saa_cos = np.cos(saa_rad)

        # Add the solar angles to the numerical data dictionary
        
        numerical_data["sza_sin"] = sza_sin
        numerical_data["sza_cos"] = sza_cos
        numerical_data["saa_sin"] = saa_sin
        numerical_data["saa_cos"] = saa_cos

        print("Data Preparation : converted to solar angles (sza, saa)")
        print("Data Preparation : returning images, numerical_data")
        return images, numerical_data

    else:

        return images, numerical_data

def extract_goes_inference(date_time, parallel_index,channels="C01",goes_aws_url_folder= 'noaa-goes16/ABI-L2-CMIPF'):
    """
    This function extracts GOES images for a given date_time and parallel_index. (whole GOES slice, used for inference which differs from images used in training that have a matched orbit with scatterometers.)
    It retrieves the GOES data from the specified AWS S3 bucket and processes it to create
    images of the specified size (128x128).

    Args:
        date_time (numpy.datetime64): The time of the GOES data.
        parallel_index (numpy.ndarray): The precomputed indices for GOES pixels corresponding to scatterometer lat/lon.
        channels (str or list): The channel(s) of interest. Default is "C01".
        goes_aws_url_folder (str): The folder in the AWS S3 bucket where the GOES data is stored. Default is 'noaa-goes16/ABI-L2-CMIPF'.
        
    Returns:
        images (list): A list of numpy arrays containing the extracted GOES images of shape (128, 128).
    """

    # ignore divide by zero errors which occur when the GOES data can't form a 128x128 image
    np.seterr(invalid='ignore', divide='ignore')

    fs = fsspec.filesystem("s3", anon=True, default_block_size=512 * 1024**1024)
    urls = get_goes_url(date_time, goes_aws_url_folder='noaa-goes16/ABI-L2-CMIPF', goes_channel= channels)
    with fs.open(urls[0], mode="rb") as f:
        print("INFO : Reading file:", urls[0])
        goes_image = xr.open_dataset(f)
        goes_image = goes_image.rename({"x": "x_index", "y": "y_index"})

        # Assign the index coordinates (if not already done)
        goes_image = goes_image.assign_coords(
            x_index=np.arange(goes_image.sizes["x_index"]),
            y_index=np.arange(goes_image.sizes["y_index"]),
        )

        images = []
        goes_image.load()
        print("INFO : Extracting images")
        for i in range(parallel_index.shape[0]):
            for j in range(parallel_index.shape[1]):
                try:
                    x_mean = parallel_index[i][j][1].mean().astype(int)
                    x_min = x_mean - 63
                    x_max = x_mean + 63
                    y_mean = parallel_index[i][j][0].mean().astype(int)
                    y_min = y_mean - 63
                    y_max = y_mean + 63
                    image = goes_image.CMI.sel(
                        x_index=slice(x_min, x_max), y_index=slice(y_min, y_max)
                    )

                    target_size = (128, 128)

                    padded_image = np.pad(
                        image,
                        (
                            (
                                (target_size[0] - image.shape[0]) // 2,
                                (target_size[0] - image.shape[0] + 1) // 2,
                            ),
                            (
                                (target_size[1] - image.shape[1]) // 2,
                                (target_size[1] - image.shape[1] + 1) // 2,
                            ),
                        ),
                        constant_values=0,
                    )

                except:
                    images.append(np.zeros((128, 128)))

                    continue
                images.append(padded_image)

        return images
