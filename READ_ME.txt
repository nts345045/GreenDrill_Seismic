Repository: GreenDrill_Seismic_Repo
Origin: 10 FEB 2023
AUTH: Nathan T. Stevens
EMAIL: nts5045@psu.edu

Purpose: Data processing and modeling scripts for active-source seismic surveying for the GreenDrill project.



:: TODO ::
End of 14. Feb 2023
Current project - projects.Prudhoe.S3_Shallow_Structure_Modeling
    Need to finish drafting driver that iterates over spread's and shot #'s
    QC/DEBUG: Currently NS02 totally fails to converge at curve_fit() and WE02 has larger uncertainties
projects.Prudhoe.S4_Ice_Thickness_Modeling
    Need to finish hyperbolic_fit - requires clean-up of Vrms estimation method
    Want to update the reflection travel-time modeling to use ray-tracing through layered media


Updated run on S1 to get CMP information into each pick's data


====Environment====
Dependencies:
 - scipy
 - numpy
 - pandas
 - obspy
 - matplotlib
 - pyrocko <- used for phase arrival picking. Can be omitted with slight modifications to t(x) and A(x) extraction codes in the projects/ subdirectories
 - ttcrpy <- used for 2-D and 3-D raytracing
 - vtk <- internal dependency for ttcrpy

Notes on Pyrocko and ttcrpy: requires a `pip` installation into a conda env. Pyrocko  may need an additional pyqt5 dependency update, depending on OS. 

====Repository Structure====
__init__.py - Make entries in this directory importable python modules
core - Core scripts with generalized arguments - building blocks for 'projects'
    __init__.py - make importable
    TimeseriesTools.py - Methods related to time and frequency domain data processing

    GeometryTools.py - Methods related to source-receiver geometry processing

    RayTracing1D.py - Methods supporting ray-tracing for modeling ray-paths with pyrocko.cake implementations of 1-D ray-tracing (Heimann and others, 2017)

    RayTracing_ttcr.py - Methods supporting ray-tracing with the ttcrpy project (Giroux, 2021)

    KirchnerBentley_Model.py - Methods for implementing the Kirchner & Bentley 
                               (1979) double-exponential travel-time vs offset model

    WiechertHerglotzBateman_Inversion - Methods supporting the vertical slowness
                                        profile inversion of W-H-B (e.g., Slichter, 1932; Riverman and others, 2019)

    Reflectivity.py - Methods related to estimating reflectivity coefficients
                      using approaches in Holland & Anandakrishnan (2009)

    Dix_Conversion.py - Methods related to Dix conversion for estimating interval
                        velocities and fitting hyperbolic t(x) data

    Firn_Density.py - Methods related to estimating the seismic density of firn using
                    methods in Riverman and others (2019), Schlegel and others (2019), Hollmann and others (2022), and references therein.

projects - Site-specific scripts for processing from raw data to final products.
    Progression of processing steps are noted by the start of each script within site-name directories (S#_ = Step #)

    __init__.py - make importable
    Prudhoe - Processing scripts for the Prudhoe Dome site
        __init__.py - make importable

        RUN_EVERYTHING.py - Run all the processing steps in order. See below for I/O & tasks.

        S1_Time_Geom_Processing.py - 
            Inputs: Pick Times, Shot Locations, Receiver Locations
            Tasks: Compile individual shot-gather picks, calculate geometries for CMP gathers, use KB79 modeling to correct non-GPS timed data (GeoRods)
            Outputs: Compiled picks with t(x) correction factors and Source-Receiver Geometry data (Compiled Picks hereafter), initial KB79 models

        S2_Amplitude_Polarity_Extraction.py - 
            Inputs: Compiled Picks, waveform file catalog (WFDISC)
            Tasks: Extract amplitude estimates from waveform data at pick times and append to compiled pick entries
            Outputs: Compiled Picks += Amplitude and polarity data on all picks, polarity agreement on all traces with direct & reflected/multiple arrivals

        S3_Vertical_Slowness_Modeling.py - 
            Inputs: Compiled picks (from S1), Initial KB79 models
            Tasks: Conduct KB79 fitting and WHB inversion (with uncertainty quant) on:
                    1) Whole data-set gather
                    2) Shot gathers
                    3) CMP gathers
                   Estimate ice-column structure using WHB profiles with reflected
                   arrival picks and Dix conversion. Quantify uncertainties
            Outputs: Gather-specific KB79 fits and vertical ice-structure models, model summary index

        S4_Firn_Density_Modeling.py - 
            Inputs: Vertical ice-structure models, parameter estimates & ops parameters
            Tasks: Create estimates of vertical firn density profiles using the range of methods implemented in `core.Firn_Density`.
            Outputs: Vertical firn density profiles
            Figures: Firn density profiles with IDP ops. casing FOS thresholds

        S5_RayTracing_Modeling.py - 
            Inputs: Vertica ice-structure models, Compiled Picks
            Tasks: Conduct ray-tracing to estimate ray-path lengths ($d_i$)
            Outputs: Ray-path-length model summary indexed by Compiled Pick

        S6_Attenuation_A0_Modeling.py - 
            Inputs: Compiled Picks, Compiled Pick Amplitudes, Compiled Pick Spectra
            Tasks: Conduct source amplitude ($A_0$), attenuation coefficient ($\alpha$), and quality factor ($Q$) estimation from A(x) and fft(A(x,t)) observations.
                Methods:
                    1) Multiple-bounce method (Holland & Anandakrishnan, 2009)
                    2) Spectral ratio method (Peters and others, 2012)
                    3) Direct, 2-station method (Holland & Anandakrishnan, 2009)
                    4) Direct-wave semilog regression (This study)
            Outputs: Summary table with indices of 

        S7_Reflectivity_Modeling - 
            Inputs: Compiled Picks, Ray-path model summary, 

    Inglefield - Processing scripts for the Inglefield Land Margin site


====References====
Abbreviations:
    A(x) - Amplitude as a function of horizontal source-receiver offset
    t(x) - travel-time as a function of horizontal source-receiver offset
    CMP - Common Mid Point
    IDP - Ice Drilling Program
    FOS - factor of safety

Annotated References:

Diez and others (2013):
    Diez, A., Eisen, O., Hofstede, C., Bohleber, P., & Polom, U. (2013). Joint interpretation of explosive and vibroseismic surveys on cold firn for the investigation of ice properties. Annals of Glaciology, 54(64). doi:10.3189/2013AoG64A200

    Notes: Explainer of WHB method, but velocity formulation is a bit odd. 

Giroux (2021):
    Giroux, B. (2021). ttcrpy: A Python package for traveltime computation and raytracing. SoftwareX, 16(100834). doi:10.1016/j.softx.2021.100834

Heimann and others (2017):
    Heimann, S., Kriegerowski, M., Isken, M., Cesca, S., Daout, S., Grigoli, F., Juretzek, C., Megies, T., Nooshiri, N., Steinberg, A., Sudhaus, H., Vasyura-Bathke, H., Willey, T., & Dahm, T. (2017): Pyrocko - An open-source seismology toolbox and library. V. 0.3. GFZ Data Services. doi:10.5880/GFZ.2.1.2017.001

    Notes: Pyrocko project - analyst picks and 1-D ray-tracing tools

Holland & Anandakrishnan (2009):
    Holland, C., & Anandakrishnan, S. (2009). Subglacial seismic reflection strategies when source amplitude and medium attenuation are poorly known. Journal of Glaciology, 55(193), 931-937. doi:10.3189/002214309790152528

    Notes: Explainer of methods for determining reflectivity.

Hollmann and others (2022): 
    Hollmann H., Treverrow A., Peters L.E., Reading A.M., Kulessa B. (2021). Seismic observations of a complex firn structure across the Amery Ice Shelf, East Antarctica. Journal of Glaciology 67(265), 777–787. doi:10.1017/jog.2021.21

    Notes: Summary of WHB and KB method + application to distributed sites.

Kirschner & Bentley (1979): 
    Kirchner, J.F., & Bentley, C.R. (1979). Seismic short-refraction studies on the Ross Ice Shelf, Antarctica. Journal of Glaciology, 24(90), 313-319.

    Notes: Initial source of double-exponential model for use in WHB analysis of
    firn. Widely used in subsequent studies.

Riverman and others (2019): 
    Riverman, K. L., Alley, R. B., Anandakrishnan, S., Christianson, K., Holschuh, N. D., Medley, B., Muto, A., & Peters, L. E. (2019). Enhanced firn densification in high‐accumulation shear margins of the NE Greenland Ice Stream. Journal of Geophysical Research: Earth Surface, 124, 365–382. doi:10.1029/2017JF004604

    Notes: Succinct WHB explainer and offers additional velocity-density conversion candidate models (polynomail) that out-perform Kohnen (1972) and Robin (19)

Schlegel and others (2019):
    Schlegel, R., Diez, A., Löwe, H., Mayer, C., Lambrecht, A., Freitag, J., Miller, H., Hofstede, C., & Eisen, O. (2019). Comparison of elastic moduli from seismic diving-wave and ice-core microstructure analysis in Antarctic polar firn. Annals of Glaciology, 60(79), 220-230. doi:10.1017/aog.2019.10

    Notes: Explores density estimation methods from seismic velocities from a compliance-matrix framework of earlier work by Diez and others (2013a,b).

Slichter (1932):
    Slichter, L.B. (1932). The theory and interpretation of seismic travel-time curves in horizontal structures. Physics, 273, 273-295. doi:10.1063/1.1745133

    Notes: Early theory paper summarizing application of the Wiechert-Herglotz-Bateman inversion.