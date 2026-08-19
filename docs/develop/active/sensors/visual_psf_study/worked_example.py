"""Worked example matching FIG 2's left column exactly: object at 20 deg, d=3.5,
range-2 diamond. Every number the page quotes is printed here."""
import numpy as np, json

TH, D, RS, ANG = 20.0, 3.5, 0.5, 10.0
th = np.radians(TH)
e = np.array([-D*np.cos(th), D*np.sin(th)])          # object offset (row, col)
u = e / np.linalg.norm(e)                            # radial unit vector
t = np.array([-u[1], u[0]])                          # tangential unit vector
sp = RS * D                                          # sigma_parallel
st = D * np.sin(np.radians(ANG))                     # sigma_perp
rho = sp / st

out = {'theta': TH, 'd': D, 'e': e.tolist(), 'u': u.tolist(), 't': t.tolist(),
       'sp': sp, 'st': st, 'rho': rho,
       'detSigma_sqrt': sp*st, 'full_mass': 2*np.pi*sp*st, 'cells': {}}

for name, c in [('N', np.array([-2., 0.])), ('E', np.array([0., 2.]))]:
    v = c - e
    vpar, vperp = float(v @ u), float(v @ t)
    ex_aniso = -vpar**2/(2*sp**2) - vperp**2/(2*st**2)
    ex_iso   = -(vpar**2 + vperp**2)/(2*sp**2)
    out['cells'][name] = {
        'cell': c.tolist(), 'v': v.tolist(), 'vpar': vpar, 'vperp': vperp,
        'term_par': -vpar**2/(2*sp**2), 'term_perp': -vperp**2/(2*st**2),
        'exp_aniso': ex_aniso, 'w_aniso': float(np.exp(ex_aniso)),
        'exp_iso': ex_iso,     'w_iso':   float(np.exp(ex_iso)),
    }
a, i = out['cells'], out['cells']
out['ratio_aniso'] = a['N']['w_aniso'] / a['E']['w_aniso']
out['ratio_iso']   = i['N']['w_iso']   / i['E']['w_iso']

# sanity: does the quadratic form equal the matrix form v^T Sigma^-1 v ?
R = np.column_stack([u, t])
Sigma = R @ np.diag([sp**2, st**2]) @ R.T
v = np.array([-2., 0.]) - e
quad_matrix = float(v @ np.linalg.inv(Sigma) @ v)
quad_explicit = (v @ u)**2/sp**2 + (v @ t)**2/st**2
out['check_quadratic_form'] = [quad_matrix, float(quad_explicit)]
out['check_det'] = [float(np.linalg.det(Sigma)), float((sp*st)**2)]

print(json.dumps(out, indent=1))
