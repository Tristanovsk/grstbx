from setuptools import setup, find_packages


__package__ = 'grstbx'
__version__ = '1.0.0'

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={
         '': ['*.nc','*.txt','*.csv','*.dat'],
         'grstbx': ['grstbx/data/*']
    },
    include_package_data=True,

    url='',
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='some snippets to play with GRS images',

    # Dependent packages (distributions)
    install_requires=['numpy < 1.23','scipy','pandas','xarray','dask','geopandas','rasterio','affine',
                      'matplotlib','docopt','pyproj','datetime','rioxarray','datetime','cartopy','folium','mapclassify',
                      'holoviews','hvplot','colorcet','panel','datashader','netcdf4','matplotlib-scalebar'
],


    # entry_points={
    #       'console_scripts': [
    #           'grstbx = TODO'
    #       ]}
)
