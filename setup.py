from setuptools import setup, find_packages


__package__ = 'grstbx'
__version__ = '2.0.2'

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
    install_requires=['scikit-learn','importlib_resources','numpy','scipy','pandas','xarray','dask','geopandas','rasterio','pandas-bokeh',
                      'matplotlib','docopt','pyproj','datetime','rioxarray','datetime','cartopy',
                      'holoviews','colorcet','panel','datashader','jupyterlab','jupyter_bokeh'],


    # entry_points={
    #       'console_scripts': [
    #           'grstbx = TODO'
    #       ]}
)
