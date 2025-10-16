import os


def Environment_Setup(environment_vars):

    # Access all parameters from the Environment_variables=environment_vars dictionary
    Computation_device = environment_vars["Computation_device"]
    GPU_ID = environment_vars["GPU_ID"]
    Gauge_group = environment_vars["Gauge_group"]
    Precision_mode = environment_vars["Precision_mode"]

    # Set computation device
    if Computation_device == 1:
        try:      
            import numba.cuda
            if numba.cuda.is_available():
                os.environ["MY_NUMBA_TARGET"] = "cuda"                                                          # Run on GPU with Numba                   
                os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
                print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (GPU parallelization)")
                print(f"Using anyone of GPUs: {GPU_ID} for computations.")
            else:
                print("CUDA (GPU) is not available. Falling back to CPU.")                
                os.environ["MY_NUMBA_TARGET"] = "numba"
                print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (CPU parallelization)")

        except ImportError:
            print("Numba/CUDA not installed. Falling back to Python.")
            os.environ["MY_NUMBA_TARGET"] = "python"
            print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (Python)")

    elif Computation_device == 2:
        os.environ["MY_NUMBA_TARGET"] = "numba"                                                                 # Run on CPU with Numba
        print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (CPU parallelization)")        

    elif Computation_device == 3:
        os.environ["MY_NUMBA_TARGET"] = "python"
        print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (Python)")

    else:
        raise ValueError("Invalid device selection: Use 1 (Numba), 2 (CUDA), or 3 (Python)")


    # Set gauge group
    if Gauge_group == 2:
        os.environ["GAUGE_GROUP"] = "su2"
        print("\nGauge group set to: SU(2)")
    elif Gauge_group == 3:
        os.environ["GAUGE_GROUP"] = "su3"
        print("\nGauge group set to: SU(3)")
    elif Gauge_group == 4:
        os.environ["GAUGE_GROUP"] = "su2_complex"
        print("\nGauge group set to: SU(2) Complex)")
    else:
        raise ValueError("\nInvalid gauge group selection: Use 1 (SU(2)) or 2 (SU(3)) or 3 (SU(2) Complex)")


    # Set precision mode of variable
    if Precision_mode == 1:
        os.environ["PRECISION"] = "single"                                                              # Single precision
        print("\nPrecision set to: Single precision (float32)")
    elif Precision_mode == 2:
        os.environ["PRECISION"] = "double"                                                              # Double precision
        print("\nPrecision set to: Double precision (float64)")
    else:
        raise ValueError("\nInvalid precision selection: Use 1 (Single) or 2 (Double)")

    return environment_vars




def Glasma_Setup(glasma_params):
    # Access all parameters from the Glasma_parameters=glasma_params dictionary

    N_Events = glasma_params["N_Events"]
    N_Sheets = glasma_params["N_Sheets"]

    L = glasma_params["L"]
    N = glasma_params["N"]
    a = glasma_params["a"]

    hbarc = glasma_params["hbarc"]
    g2mu = glasma_params["g2mu"]
    g = glasma_params["g"]
    mu = glasma_params["mu"]    
    mu_Latt = glasma_params["mu_Latt"]  

    m_IR = glasma_params["m_IR"]
    m_UV = glasma_params["m_UV"]
    m_IR_Latt = glasma_params["m_IR_Latt"]  
    m_UV_Latt = glasma_params["m_UV_Latt"]  

    tau = glasma_params["tau"]
    tau_final_Phys = glasma_params["tau_final_Phys"]
    tau_final_Latt = glasma_params["tau_final_Latt"]
    dt = glasma_params["dt"]
    N_timesteps_per_latt_spacing = glasma_params["N_timesteps_per_latt_spacing"]
    N_total_time_steps = glasma_params["N_total_time_steps"]

    GeV_to_Latt = glasma_params["GeV_to_Latt"]
    fm_to_Latt = glasma_params["fm_to_Latt"]
    Latt_to_GeV = glasma_params["Latt_to_GeV"]
    Latt_to_fm = glasma_params["Latt_to_fm"]

    Lypnv_alpha              =  glasma_params["Ly_alpha"]
    Lypnv_noise_scale        =  glasma_params["Ly_noise_scale"]
    Lypnv_noise_Phys         =  glasma_params["Ly_noise_Phys"]
    Lypnv_noise_Latt         =  glasma_params["Ly_noise_Latt"]
    Option_Lypnv_noise_type  =  glasma_params["Option_Ly_noise_type"]


    if Option_Lypnv_noise_type == 0:
        Lypnv_Noise_type = "No noise"
        Lypnv_Noise_type_str = "None"

    elif Option_Lypnv_noise_type == 1:
        Lypnv_Noise_type = "Exponential noise"
        Lypnv_Noise_type_str = "EXP"
 
    elif Option_Lypnv_noise_type == 2:
        Lypnv_Noise_type = "Power-law noise"
        Lypnv_Noise_type_str = "PL"       
    else:
        raise ValueError("\nInvalid Lyapunov noise type selection: Use 0 (No noise), 1 (Exponential noise) or 2 (Power-law noise)")
    

    Lypnv_alpha_str = f"{Lypnv_alpha:.6f}".rstrip('0').rstrip('.')
    Lypnv_mnoise_str = f"{Lypnv_noise_scale:.1f}" 


    
    print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " )
    print("+++++++++++++++++++++++++++++++++++++ PARAMETERS USED FOR GLASMA: +++++++++++++++++++++++++++++++++++++ ") 
    print("No. of events, N_Events  = ", N_Events)   
    print("No. of sheets, N_Sheets  = ", N_Sheets, "\n")   
   
    #print("Radius of the nucleus, RA          = ", RA, "fm")
    #print("Transverse area of the nucleus, F  = ", F, "fm^2", "\n")
    print("Length of the simulation box, L_tranverse  = ", L, "fm")
    print("Number of lattice points, N_LatticePoints  = ", N)
    print("Lattice spacing, a = delta_x               = ", a, "fm", "\n")

    print("hbarc                 = ", hbarc, "GeV-fm", "\n")
    print("g2mu                  = ", g2mu, "GeV")
    print("Coupling constant, g  = ", g)
    print("mu                    = ", mu, "GeV")
    print("mu (in dimensionless units), mu_Latt = ", mu_Latt, "lattice units", "\n")


    print("Infra-red cutoff,     m_IR  = ", m_IR, "GeV")
    print("Ultra-violet cutoff,  m_UV  = ", m_UV, "GeV")
    print("Infra-red cutoff    (in dimensionless units),  m_IR_Latt = ", m_IR_Latt, "lattice units")
    print("Ultra-violet cutoff (in dimensionless units),  m_UV_Latt = ", m_UV_Latt, "lattice units", "\n")




    print("GeV to lattice units, GeV_to_Latt  = ", GeV_to_Latt)
    print("fm to lattice units, fm_to_Latt    = ", fm_to_Latt)
    print("lattice units to GeV, Latt_to_GeV  = ", Latt_to_GeV)
    print("lattice units to fm, Latt_to_fm    = ", Latt_to_fm, "\n")

    print("tau  = ", tau)
    print("Maximum proper time (in physical units), tau_final_Phys       = ", tau_final_Phys, "fm/c or fm")
    print("Maximum proper time (in dimensionless units), tau_final_Latt  = ", tau_final_Latt, "lattice units", "\n")
    print("Number of time steps per transverse spacing, N_timesteps_per_latt_spacing  = ", N_timesteps_per_latt_spacing)
    print("Total No. of time steps for full Glasma simulation, N_total_time_steps     = ", N_total_time_steps, "\n")
    print("1/N_timesteps_per_latt_spacing, dt  = ", dt)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " )

    print("\n======================================================================================================= ")
    print("============================== PARAMETERS USED FOR LYPANUNOV EXPONENTS: =============================== ") 
    print("Lypnv_alpha         = ", Lypnv_alpha, "\n")
    print("Lypnv_noise_scale   = ", Lypnv_noise_scale, "\n")   

    print("Lypnv_noise_Phys    = ", Lypnv_noise_Phys, "GeV")                                 
    print("Lypnv_noise_Latt    = ", Lypnv_noise_Latt, "lattice units", "\n")

    print("Lypnv_Noise_typ     = ", Lypnv_Noise_type, "\n")    

    print("Noise type string  = ", Lypnv_Noise_type_str)
    print("alpha string       = ", Lypnv_alpha_str)
    print("mnoise string      = ", Lypnv_mnoise_str)
    print("======================================================================================================= ", "\n")

   


    return glasma_params