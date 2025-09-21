
"""
geostrophic_wind_demo.py

Compute geostrophic wind from a pressure (Pa) or height (m) field on a lat/lon grid.

Usage (as a module):
    from geostrophic_wind_demo import geostrophic_from_pressure, geostrophic_from_height

This file also includes a __main__ section that runs a synthetic example:
    python3 geostrophic_wind_demo.py
"""

import numpy as np

OMEGA = 7.2921159e-5  # s^-1
R_EARTH = 6_371_000.0 # m
G = 9.80665           # m s^-2

def coriolis(lat):
    """Coriolis parameter f = 2 * Omega * sin(lat). lat in degrees, returns 2D array if lat is 2D."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat))

def metric_dx_dy(lat, dlon_deg, dlat_deg):
    """
    Return grid spacing in meters for given latitude (deg) and grid step (deg).
    dx = R * cos(phi) * dlon, dy = R * dlat, where angles are radians.
    Accepts lat as a 2D array to produce 2D dx, dy.
    """
    dlon = np.deg2rad(dlon_deg)
    dlat = np.deg2rad(dlat_deg)
    dx = R_EARTH * np.cos(np.deg2rad(lat)) * dlon
    dy = R_EARTH * dlat * np.ones_like(lat)
    return dx, dy

def centered_gradient(field, dx, dy):
    """
    Compute centered finite-difference gradients d(field)/dx and d(field)/dy.
    field: 2D array (ny, nx)
    dx, dy: 2D arrays (ny, nx) of grid spacing in meters.
    Edges use one-sided differences.
    Returns dfdx, dfdy (same shape).
    """
    ny, nx = field.shape
    dfdx = np.empty_like(field, dtype=float)
    dfdy = np.empty_like(field, dtype=float)

    # X direction
    dfdx[:,1:-1] = (field[:,2:] - field[:, :-2]) / (0.5*(dx[:,2:] + dx[:, :-2]))
    dfdx[:,0]    = (field[:,1]  - field[:,0])   / dx[:,0]
    dfdx[:,-1]   = (field[:,-1] - field[:,-2])  / dx[:,-1]

    # Y direction
    dfdy[1:-1,:] = (field[2:,:] - field[:-2,:]) / (0.5*(dy[2:,:] + dy[:-2,:]))
    dfdy[0,:]    = (field[1,:]  - field[0,:])   / dy[0,:]
    dfdy[-1,:]   = (field[-1,:] - field[-2,:])  / dy[-1,:]

    return dfdx, dfdy

def geostrophic_from_pressure(p, lat, dlon_deg, dlat_deg, rho=1.225, f_min=1e-5):
    """
    Compute geostrophic wind (u_g, v_g) from pressure (Pa) on a constant-height surface.
    u_g = - (1/(rho f)) * dp/dy
    v_g =   (1/(rho f)) * dp/dx
    Inputs:
        p:   2D array pressure in Pa, shape (ny, nx)
        lat: 2D array of latitudes (deg), same shape as p (or 1D broadcastable)
        dlon_deg, dlat_deg: grid spacing in degrees (scalars)
        rho: air density (kg m^-3), default ~ sea level
        f_min: minimum |f| to avoid blowup near equator
    Returns: u_g, v_g in m/s
    """
    lat2d = np.broadcast_to(lat, p.shape)
    dx, dy = metric_dx_dy(lat2d, dlon_deg, dlat_deg)
    dpx, dpy = centered_gradient(p, dx, dy)
    f = coriolis(lat2d)
    f = np.where(np.abs(f) < f_min, np.sign(f)*f_min + (f==0)*f_min, f)  # clip small f
    ug = - (1.0/(rho * f)) * dpy
    vg =   (1.0/(rho * f)) * dpx
    return ug, vg

def geostrophic_from_height(Z, lat, dlon_deg, dlat_deg, f_min=1e-5):
    """
    Compute geostrophic wind (u_g, v_g) from geopotential height (m) on an isobaric surface.
    u_g = - (g/f) * dZ/dy
    v_g =   (g/f) * dZ/dx
    """
    lat2d = np.broadcast_to(lat, Z.shape)
    dx, dy = metric_dx_dy(lat2d, dlon_deg, dlat_deg)
    dZdx, dZdy = centered_gradient(Z, dx, dy)
    f = coriolis(lat2d)
    f = np.where(np.abs(f) < f_min, np.sign(f)*f_min + (f==0)*f_min, f)
    ug = - (G / f) * dZdy
    vg =   (G / f) * dZdx
    return ug, vg

def synthetic_pressure(lat, lon, p0=101325.0, amp=2000.0, lat0=-35.0, lon0=145.0, scale_km=800.0):
    """
    A smooth low-pressure 'vortex-like' Gaussian in Pa centered at (lat0, lon0).
    """
    # great-circle approx into meters
    dphi = np.deg2rad(lat - lat0)
    dlmb = np.deg2rad(lon - lon0)
    # approximate local metric using mean latitude
    latm = 0.5*(lat + lat0)
    dx = R_EARTH * np.cos(np.deg2rad(latm)) * dlmb
    dy = R_EARTH * dphi
    r2 = dx*dx + dy*dy
    s2 = (scale_km*1000.0)**2
    return p0 - amp*np.exp(-r2/(2*s2))

if __name__ == "__main__":
    # Grid over southeastern Australia (roughly)
    ny, nx = 60, 80
    lat = np.linspace(-45, -25, ny)[:,None] * np.ones((1,nx))
    lon = np.ones((ny,1)) * np.linspace(135, 155, nx)[None,:]

    p = synthetic_pressure(lat, lon, lat0=-37.8, lon0=144.9, scale_km=700.0)
    ug, vg = geostrophic_from_pressure(p, lat, dlon_deg=(lon[0,1]-lon[0,0]), dlat_deg=(lat[1,0]-lat[0,0]))

    # Save small sample outputs for inspection
    np.savez("geostrophic_synthetic_example.npz", lat=lat, lon=lon, p=p, ug=ug, vg=vg)
    print("Wrote geostrophic_synthetic_example.npz")
