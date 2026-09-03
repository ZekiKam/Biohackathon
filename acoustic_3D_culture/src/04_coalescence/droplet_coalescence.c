/**
 * droplet_coalescence.c — Basilisk simulation
 * ============================================
 * Two-phase Navier–Stokes with surface tension (Volume-of-Fluid method).
 * Models the coalescence of two water droplets in air after they reach
 * the acoustic trap and come into contact.
 *
 * Physics
 * -------
 *   ρ (∂u/∂t + u·∇u) = −∇p + ∇·(μ D) + σ κ δ_s n̂
 *   ∂f/∂t + u·∇f = 0                (VOF advection)
 *   ρ = f ρ_water + (1−f) ρ_air
 *   μ = f μ_water + (1−f) μ_air
 *
 * Where f ∈ [0,1] is the water volume fraction, σ is surface tension,
 * κ is interface curvature, and δ_s is a surface-concentrated force.
 *
 * Initial conditions are read from  ../output/impact_conditions.txt
 * (written by Stage 3's postprocess_for_basilisk.py).
 *
 * Compile (inside WSL 2, with Basilisk installed — see README_wsl.md):
 *   qcc -O2 -Wall droplet_coalescence.c -o coalescence -lm
 *
 * Run:
 *   ./coalescence
 *
 * Outputs:
 *   snapshots/snapshot-*.vtk   (ParaView-ready VTK files)
 *   log.csv                    (global diagnostics vs time)
 *
 * Reference: Popinet, J. Comput. Phys. 228, 5838 (2009)
 */

#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"
#include "view.h"

/* -------------------------------------------------------------------------
   Physical parameters  (SI units)
   ------------------------------------------------------------------------- */
#define RHO_WATER    1000.0    /* kg/m³ */
#define RHO_AIR      1.204     /* kg/m³ */
#define MU_WATER     1.0e-3    /* Pa·s  */
#define MU_AIR       1.81e-5   /* Pa·s  */
#define SIGMA_SURF   0.072     /* N/m   — water–air surface tension */

/* Droplet radius [m] — should match config.py A_DROP */
#define A_DROP       1.0e-3

/* Domain half-size: 8 droplet diameters */
#define DOMAIN       (8.0 * A_DROP)

/* Impact velocity  [m/s]  — overridden at runtime from impact_conditions.txt */
static double v_impact  = 5e-4;    /* default: 0.5 mm/s */
static double b_param   = 0.0;     /* impact parameter (0 = head-on) */

/* Simulation end time  [s] */
#define T_END        (20.0 * A_DROP / v_impact)   /* ~20 crossing times */

/* Maximum adaptive mesh refinement level */
#define MAXLEVEL     8

/* Snapshot interval */
#define DT_SNAP      (T_END / 200.0)

/* -------------------------------------------------------------------------
   Helper: initialise two spherical droplets
   Droplet 1 centred at (-sep/2, 0, 0), moving in +x
   Droplet 2 centred at (+sep/2, 0, 0), moving in -x
   where sep = 2*A_DROP + small_gap
   ------------------------------------------------------------------------- */
static double separation;   /* set in main() */

double droplet_fraction (double x, double y, double z,
                         double cx, double cy, double cz, double R)
{
    double r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
    /* Smooth step across the interface at radius R over ≈ Δ/2 */
    double eps = DOMAIN / (1 << MAXLEVEL);   /* grid spacing at finest level */
    double d   = sqrt(r2) - R;
    if (d < -eps) return 1.0;
    if (d >  eps) return 0.0;
    return 0.5 - 0.5 * sin(M_PI * d / (2.0 * eps));
}

/* -------------------------------------------------------------------------
   Basilisk event hooks
   ------------------------------------------------------------------------- */

int main (int argc, char * argv[])
{
    /* Optionally read impact conditions from file */
    FILE * ic = fopen ("../output/impact_conditions.txt", "r");
    if (ic) {
        double vi, bi;
        if (fscanf (ic, "%lf %lf", &vi, &bi) == 2) {
            v_impact = vi;
            b_param  = bi;
        }
        fclose (ic);
    }

    separation = 2.0 * A_DROP * 1.01;   /* tiny gap between droplets */

    /* Fluid properties */
    rho1 = RHO_WATER;   rho2 = RHO_AIR;
    mu1  = MU_WATER;    mu2  = MU_AIR;
    f.sigma = SIGMA_SURF;

    /* Adaptive Cartesian grid */
    init_grid (1 << 6);
    size (2.0 * DOMAIN);
    origin (-DOMAIN, -DOMAIN, -DOMAIN);

    run();
}

event init (t = 0)
{
    /* Initialise volume fraction field */
    fraction (f, (droplet_fraction(x,y,z,
                                   -separation/2.0, b_param*A_DROP, 0.0,
                                   A_DROP)
               + droplet_fraction(x,y,z,
                                   +separation/2.0, 0.0, 0.0,
                                   A_DROP)
               - 1.0));   /* union of two spheres */

    /* Initial velocity field: droplets approaching along x */
    foreach() {
        double f_left  = droplet_fraction(x,y,z, -separation/2.0, b_param*A_DROP, 0.0, A_DROP);
        double f_right = droplet_fraction(x,y,z, +separation/2.0, 0.0, 0.0, A_DROP);
        u.x[] = f_left * (+v_impact) + f_right * (-v_impact);
        u.y[] = 0.0;
        u.z[] = 0.0;
    }
}

/* Adaptive mesh refinement — refine near the interface */
event adapt (i++)
{
    adapt_wavelet ({f, u.x, u.y},
                   (double[]){0.01, 1e-3, 1e-3},
                   maxlevel=MAXLEVEL, minlevel=4);
}

/* Snapshots saved as VTK */
event snapshots (t = 0.0; t += DT_SNAP; t <= T_END)
{
    static int n = 0;
    char fname[128];

    /* Write f (volume fraction) and |u| as VTK */
    snprintf (fname, sizeof(fname), "snapshots/snapshot-%04d.vtk", n++);
    FILE * fp = fopen (fname, "w");
    if (!fp) {
        /* Try creating directory and retry */
        system ("mkdir -p snapshots");
        fp = fopen (fname, "w");
    }
    if (fp) {
        output_vtk ({f, u.x, u.y, u.z}, N, fp, false);
        fclose (fp);
    }
}

/* CSV log: time, droplet volume, kinetic energy, surface energy */
event logfile (i++)
{
    double vol = 0.0, Ek = 0.0;
    foreach (reduction(+:vol) reduction(+:Ek)) {
        vol += f[]    * dv();
        Ek  += 0.5 * rho[] * (sq(u.x[]) + sq(u.y[])) * dv();
    }
    static bool header = true;
    FILE * log = fopen ("log.csv", header ? "w" : "a");
    if (header) { fprintf (log, "t,volume,kinetic_energy\n"); header = false; }
    fprintf (log, "%.6e,%.6e,%.6e\n", t, vol, Ek);
    fclose (log);
}

event end (t = T_END) {}
