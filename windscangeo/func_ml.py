import datetime
import wandb
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import torch
import torch.utils.data
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

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

class Normalize:

    """
    Normalize the input tensor by subtracting the mean and dividing by the standard deviation. Done per batch 

    Args:
        mean (list or np.ndarray): Mean values for normalization.
        std (list or np.ndarray): Standard deviation values for normalization.
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std
    
class H5pyDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading data from an HDF5 file. This is useful when dealing with large datasets that do not fit into memory.
    Need to work on Zarr integration for better performance

    Args:
        h5_file_path (str): Path to the HDF5 file.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, h5_file_path, transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.file = None  # Will be initialized per worker
        with h5py.File(self.h5_file_path, 'r') as f:
            self.length = len(f['targets'])

    def _ensure_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_file_path, 'r')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_file()

        image = self.file['images'][idx]
        target = self.file['targets'][idx]

        image = torch.tensor(image, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target

    def __del__(self):
        if self.file:
            self.file.close()
    
class conventional_dataset_inference(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading data for inference (no lable) using regular numpy arrays.

    Args:
        images (list or np.ndarray): List or array of images to be used for inference.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, images,transform=None):
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # 1) Image
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        return image
    
class conventional_dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading data using regular numpy arrays.

    Args:
        images (list or np.ndarray): List or array of images.
        targets (list or np.ndarray): List or array of targets corresponding to the images.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # 1) Image
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        # 2) Target for sample "idx"
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, target
    
def error_plot(best_val_outputs, best_val_labels, path_folder=None):
    """
    Plot a scatter plot of model outputs vs true labels for the validation dataset.

    Args:
        best_val_outputs (list or np.ndarray): Model outputs for the validation dataset.
        best_val_labels (list or np.ndarray): True labels for the validation dataset.
        path_folder (str, optional): Path to save the plot. If None, the plot will not be saved.
    """

    max_all = max(max(best_val_outputs), max(best_val_labels))

    plt.figure(figsize=(5, 5))
    plt.plot(best_val_labels, best_val_outputs, "o")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, max_all)
    plt.ylim(0, max_all)
    plt.xlabel("True Labels")
    plt.ylabel("Model Output")
    plt.title("Model Output vs True Labels in test dataset")
    plt.xticks(np.arange(0, 30, 5))
    plt.yticks(np.arange(0, 30, 5))
    plt.plot(
        [min(best_val_labels), max(best_val_labels)],
        [min(best_val_labels), max(best_val_labels)],
        "r--",
    )  # y = x reference line
    if path_folder:
        plt.savefig(os.path.join(path_folder, "scatter_plot.png"))


def plot_save_loss(
    best_val_outputs,
    best_val_labels,
    train_losses,
    val_losses,
    path_folder,
    saving=False,
):
    """
    Plot and save the training and validation losses, and optionally save the best validation outputs and labels.

    Args:
        best_val_outputs (list or np.ndarray): Model outputs for the validation dataset.
        best_val_labels (list or np.ndarray): True labels for the validation dataset.
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        path_folder (str): Path to save the plot and optionally the outputs and labels.
        saving (bool, optional): If True, save the best validation outputs and labels. Default is False.
    """
    # After training, save only the best validation outputs and labels
    if saving:
        np.save(
            os.path.join(path_folder, "best_validation_outputs.npy"), best_val_outputs
        )
        np.save(
            os.path.join(path_folder, "best_validation_labels.npy"), best_val_labels
        )

    num_epochs = len(train_losses)
    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    text_str = f"num_epochs = {num_epochs}, train loss = {train_losses[-1]:.2f}, validation loss = {val_losses[-1]:.2f}"
    plt.text(
        0.05,
        0.05,
        text_str,
        ha="left",
        va="bottom",
        transform=plt.gca().transAxes,  # Ensures the coordinates are relative to the axes (0 to 1 range)
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_folder, "loss_plot.png"))




def rmse_per_range(model_output, target,path_folder):
    """
    Calculate the RMSE for different ranges of wind speeds and save the results to a CSV file.

    Args:
        model_output (list or np.ndarray): Model outputs for the validation dataset.
        target (list or np.ndarray): True labels for the validation dataset.
        path_folder (str): Path to save the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the RMSE and count for each range.
    """

    max_target = np.max(target)
    bins = np.arange(0, max_target, 1)
    rmse = np.zeros(len(bins))
    count = np.zeros(len(bins))
    results = []

    for i in range(len(bins)-1):
        idx = np.where((target >= bins[i]) & (target <= bins[i+1]))
        rmse[i] = np.sqrt(np.mean((model_output[idx] - target[idx])**2))
        count[i] = len(idx[0])
        print(f"EVAL : Range {bins[i]} m/s - {bins[i+1]} m/s: RMSE = {rmse[i]}, count = {int(count[i])}")
        results.append({'bin_start': bins[i], 'bin_end': bins[i+1], 'rmse': rmse[i], 'count': count[i]})
        
    df = pd.DataFrame(results)
    df.to_csv(f'{path_folder}/rmse_per_range.csv')
    return df


def plot_cloud_mask(lat_inference,lon_inference,wind_speeds_inference,path_folder,buoy_name,buoy_lat,buoy_lon,time_choice):
    """
    Plot the cloud mask on a map with buoy locations. This works well with 30x30 images but larger images can diluted. Can be adapted to work with larger images.
    
    Args:
        lat_inference (np.ndarray): Latitude values for the inference grid.
        lon_inference (np.ndarray): Longitude values for the inference grid.
        wind_speeds_inference (np.ndarray): Cloud mask data to be plotted.
        path_folder (str): Path to save the plot.
        buoy_name (list or np.ndarray): Names of the buoys.
        buoy_lat (list or np.ndarray): Latitude values of the buoys.
        buoy_lon (list or np.ndarray): Longitude values of the buoys.
        time_choice (datetime): Time of the inference.
    """

    fig = plt.figure(figsize=(20, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot wind speed data (continuous colormap)
    pcm = ax.pcolormesh(
        lon_inference, lat_inference, wind_speeds_inference,
        shading='auto', cmap='Blues',
        vmin=0, vmax=1
    )

    # Add colorbar
    cbar = fig.colorbar(pcm, label='0 = Clear, 1 = Cloudy')

    # Add coastlines and land
    ax.add_feature(cfeature.LAND, color='white', alpha=1, zorder=10)  
    ax.coastlines(zorder=11)

    # Flatten buoy_name if it's a list of arrays
    buoy_name_flat = np.concatenate(buoy_name).tolist()
    unique_buoys = list(set(buoy_name_flat))

    # Generate enough distinct colors for all buoys from the "tab20" palette
    # (tab20 provides 20 colors; if there are more than 20 unique buoys, colors will repeat)
    color_map = plt.cm.get_cmap("tab20", len(unique_buoys))

    # Plot each buoy in a single color
    for i, buoy_id in enumerate(unique_buoys):
        # Pick a distinct color from tab20
        color = color_map(i)

        # Identify the indices belonging to this buoy
        mask = np.array(buoy_name_flat) == buoy_id

        # Scatter just those points
        ax.scatter(
            np.array(buoy_lon)[mask],
            np.array(buoy_lat)[mask],
            s=100,
            color=color,            # Set the fill color
            edgecolor='black',
            linewidth=1,
            zorder=12,
            label=buoy_id           # Use buoy_id as the legend label
        )

    # Create the legend
    leg = ax.legend(
        title="Buoy Stations",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=ax.transAxes
    )

    # Ensure legend is above all other layers
    leg.set_zorder(999)

    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False

    ax.set_title(f'Cloud Mask at time {time_choice}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_ylim(lat_inference.min(), lat_inference.max())
    ax.set_xlim(lon_inference.min(), lon_inference.max())


    plt.savefig(f'{path_folder}/plot_cloud_mask.png')



def plot_goes_image(lat_inference,lon_inference,images,path_folder,buoy_name,buoy_lat,buoy_lon,time_choice):
    """
    Plot the GOES image data on a map with buoy locations. Made for 128x128 images where the middpoint is at (64,64). If using other image sizes, the plotting will probably not work as expected.

    Args:
        lat_inference (np.ndarray): Latitude values for the inference grid.
        lon_inference (np.ndarray): Longitude values for the inference grid.
        images (np.ndarray): GOES image data to be plotted.
        path_folder (str): Path to save the plot.
        buoy_name (list or np.ndarray): Names of the buoys.
        buoy_lat (list or np.ndarray): Latitude values of the buoys.
        buoy_lon (list or np.ndarray): Longitude values of the buoys.
        time_choice (datetime): Time of the inference.
    """
    mean_images = images[:,:,64,64]
    mean_images = mean_images.ravel()
    mean_images = mean_images.reshape(160,340)
    

    fig = plt.figure(figsize=(20, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())


    # Plot wind speed data (continuous colormap)
    pcm = ax.pcolormesh(
        lon_inference, lat_inference, mean_images,
        shading='auto',
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = fig.colorbar(pcm, label='Brightness Temperature (K) - C01')

    # Add coastlines and land
    ax.add_feature(cfeature.LAND, color='white', alpha=1, zorder=10)  
    ax.coastlines(zorder=11)
    

    # Flatten buoy_name if it's a list of arrays
    buoy_name_flat = np.concatenate(buoy_name).tolist()
    unique_buoys = list(set(buoy_name_flat))

    # Generate enough distinct colors for all buoys from the "tab20" palette
    # (tab20 provides 20 colors; if there are more than 20 unique buoys, colors will repeat)
    color_map = plt.cm.get_cmap("tab20", len(unique_buoys))


    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False

    ax.set_title(f'GOES image at time {time_choice}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_ylim(lat_inference.min(), lat_inference.max())
    ax.set_xlim(lon_inference.min(), lon_inference.max())

    plt.savefig(f'{path_folder}/plot_goes_image.png')

    mean_images = np.array(mean_images)

    return mean_images


def plot_cloud_cover(lat_inference,lon_inference,images,path_folder,buoy_name,buoy_lat,buoy_lon,time_choice):
    """
    Plot the cloud cover mask on a map with buoy locations. This works well with 30x30 images but larger images can diluted. Can be adapted to work with larger images.

    Args:
        lat_inference (np.ndarray): Latitude values for the inference grid.
        lon_inference (np.ndarray): Longitude values for the inference grid.
        images (np.ndarray): GOES image data to be used for cloud cover calculation.
        path_folder (str): Path to save the plot.
        buoy_name (list or np.ndarray): Names of the buoys.
        buoy_lat (list or np.ndarray): Latitude values of the buoys.
        buoy_lon (list or np.ndarray): Longitude values of the buoys.
        time_choice (datetime): Time of the inference.
    """
    mean_images = np.mean(images, axis=(2,3))
    threshold = 0.11
    cloud_mask = np.where(mean_images > threshold, 1, 0)
    cloud_mask = cloud_mask.reshape(160,340)

    plot_cloud_mask(lat_inference,lon_inference,cloud_mask,path_folder,buoy_name,buoy_lat,buoy_lon,time_choice)

    percentage_cloud = np.sum(cloud_mask)/cloud_mask.size
    print('Cloud coverage : ',percentage_cloud*100,'%')

    return cloud_mask, percentage_cloud
    # find buoy under cloud


def plot_wind_speeds(lat_inference,lon_inference,wind_speeds_inference,path_folder,buoy_name,buoy_lat,buoy_lon,time_choice):
    """
    Plot the wind speeds on a map with buoy locations. Filter nighttime images and add coastlines, gridlines, and buoy locations.

    Args:
        lat_inference (np.ndarray): Latitude values for the inference grid.
        lon_inference (np.ndarray): Longitude values for the inference grid.
        wind_speeds_inference (np.ndarray): Wind speed data to be plotted.
        path_folder (str): Path to save the plot.
        buoy_name (list or np.ndarray): Names of the buoys.
        buoy_lat (list or np.ndarray): Latitude values of the buoys.
        buoy_lon (list or np.ndarray): Longitude values of the buoys.
        time_choice (datetime): Time of the inference.
    """
    time_str = time_choice.strftime('%Y-%m-%d %H:%M:%S')
    date, time = time_str.split(' ')
    lat = lat_inference
    lon = lon_inference
    wind_speeds = wind_speeds_inference


    buoy_names = buoy_name
    buoy_lat = buoy_lat
    buoy_lon = buoy_lon


    # nighttime mask

    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    time_flat = np.full(len(lat_flat), pd.Timestamp(f'{date} {time}'), dtype='datetime64[ns]')


    sza, saa = vectorized_solar_angles(lat_flat, lon_flat, time_flat)
    saa = np.reshape(saa,lat.shape)
    sza = np.reshape(sza,lat.shape)

    night_time_mask = np.where(sza > 90, 1, np.nan)
    cmap = ListedColormap(['white'])

    ###############

    min_lon, max_lon, min_lat, max_lat = -70, 0, -12, 20

    # add coastlines and gridlines
    fig = plt.figure(figsize=(22, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    levels = np.arange(0, 13, 1)
    line_colour = 'black'
    line_colours = ['black' for i in levels]
    ax.title.set_text(f'Wind Speed prediction from C01 GOES image (m/s) {date} {time} ')
    ax.pcolormesh(lon, lat, wind_speeds, transform=ccrs.PlateCarree(), cmap='jet',alpha=0.6,vmin=0,vmax=15,zorder = 5)
    ax.contourf(lon, lat, night_time_mask, transform=ccrs.PlateCarree(),cmap=cmap,zorder = 10)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black',zorder= 20)
    ax.set_xticks(np.arange(-70, 1, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}°E'))
    ax.set_yticks(np.arange(-15, 21, 5), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}°N'))
    ax.set_extent((-70, 0, -12, 20))
    ax.hlines(0, -70, 0, color='red', linewidth=1.5, linestyle='--', zorder= 10)
    fig.colorbar(ax.pcolormesh(lon, lat, wind_speeds, transform=ccrs.PlateCarree(), cmap='jet',alpha=0.6,vmin=0,vmax=15), ax=ax, orientation='vertical', aspect=50, label='Wind Speed (m/s)')
    ax.gridlines(color='gray', linestyle='--', alpha=0.5,zorder= 999)

    for buoy in range(len(buoy_names)):
        lon_b, lat_b = buoy_lon[buoy], buoy_lat[buoy]
        
        # Skip if out of bounds
        if not (min_lon <= lon_b <= max_lon and min_lat <= lat_b <= max_lat):
            continue

        ax.plot(lon_b, lat_b, 'o', color='red', markersize=5, transform=ccrs.PlateCarree(), label=buoy_names[buoy],zorder = 999)

        ax.text(
            lon_b + 0.2, lat_b + 0.2, buoy_names[buoy],
            fontsize=9, color='white',
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='black', edgecolor='none', boxstyle='square,pad=0.2'),zorder = 999
        )
    plt.savefig(f'{path_folder}/plot_wind_speeds.png')



