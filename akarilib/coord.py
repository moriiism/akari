from astropy.coordinates import SkyCoord
from astropy import units

def ecliptic_to_radec(ecliptic_lon,
                      ecliptic_lat):
    coord = SkyCoord(
        lon=ecliptic_lon, lat=ecliptic_lat,
        frame='barycentricmeanecliptic',
        unit=(units.degree, units.degree)).transform_to('icrs')
    ra = coord.ra.degree
    dec = coord.dec.degree
    return [ra, dec]

def radec_to_ecliptic(ra, dec):
    ecliptic = SkyCoord(ra, dec,
                        unit=(units.degree, units.degree),
                        frame='icrs').transform_to(
                            'barycentricmeanecliptic')
    ecliptic_lon = ecliptic.lon.degree
    ecliptic_lat = ecliptic.lat.degree
    return [ecliptic_lon,
            ecliptic_lat]
