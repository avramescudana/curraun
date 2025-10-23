import os
from numba import set_num_threads, get_num_threads


def Environment_Setup(environment_vars):

    # Access all parameters from the Environment_variables=environment_vars dictionary
    Computation_device   =  environment_vars["Computation_device"]
    GPU_ID               =  environment_vars["GPU_ID"]
    CPU_NumberofThreads  =  environment_vars["CPU_NumberofThreads"]

    Gauge_group          =  environment_vars["Gauge_group"]
    Precision_mode       =  environment_vars["Precision_mode"]



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

                set_num_threads(int(CPU_NumberofThreads))    
                print("No. of NUMBA threads used by CPU:", get_num_threads())  


        except ImportError:
            print("Numba/CUDA not installed. Falling back to Python.")
            os.environ["MY_NUMBA_TARGET"] = "python"
            print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (Python)")

    elif Computation_device == 2:
        os.environ["MY_NUMBA_TARGET"] = "numba"                                                                 # Run on CPU with Numba
        print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (CPU parallelization)")            
        
        set_num_threads(int(CPU_NumberofThreads))
        print("No. of NUMBA threads used by CPU:", get_num_threads())  


        # Uncomment these lines to limit number of threads
        #os.environ["OMP_NUM_THREADS"] = "4"         # OpenMP        # OpenMP: Used by NumPy, SciPy, and some C extensions
        #os.environ["MKL_NUM_THREADS"] = "4"         # Intel MKL     # Intel MKL: Accelerates linear algebra in NumPy/SciPy (if installed).
        #os.environ["OPENBLAS_NUM_THREADS"] = "4"    # OpenBLAS      # OpenBLAS: Alternative to MKL for matrix operations.
        #os.environ["NUMBA_NUM_THREADS"] = "4"       # Numba (alternative to set_num_threads) # Numba: Parallel CPU execution (e.g., @njit(parallel=True)).




        # ======================= VERIFICATION LINES FOR NO. OF THREADS =======================
        # from numba import get_num_threads
        # print("\nNumba threads:", get_num_threads())              # Should output the number given here: os.environ["NUMBA_NUM_THREADS"] = "4" 

        # ????????????????????????????????? Doesn't work ????????????????????????????????? 
        #print("NumPy max threads:", np.__config__.show())  # Should reflect your limit
        #print("NumPy max threads:", np.__config__.get_info("max_threads"))  # Should reflect your limit     # The error occurs because np.__config__.get_info() doesn't exist in newer NumPy versions. 

        # New way to check NumPy threading (works for all NumPy versions)
        #try:
        #    from threadpoolctl import threadpool_info
        #    for lib in threadpool_info():
        #        if lib['internal_api'] in ('openmp', 'mkl', 'openblas'):
        #            print(f"{lib['internal_api'].upper()} threads:", lib['num_threads'])
        #except ImportError:
        #    print("Install threadpoolctl for detailed thread info: pip install threadpoolctl")
        #    print("NumPy thread config:", np.__config__.show())

# =====================================================================================

    elif Computation_device == 3:
        os.environ["MY_NUMBA_TARGET"] = "python"
        print(f"Computation device set to: {os.environ['MY_NUMBA_TARGET']} (Python)")

    else:
        raise ValueError("Invalid device selection: Use 1 (Numba), 2 (CUDA), or 3 (Python)")







    # Set gauge group
    if Gauge_group == 1:
        os.environ["GAUGE_GROUP"]  = "su2_complex"
        Gauge_Group_String         = "SU2_Complex"
        print("\nGauge group set to:",  Gauge_Group_String)


    if Gauge_group == 2:
        os.environ["GAUGE_GROUP"]  = "su2"
        Gauge_Group_String         = "SU2"
        print("\nGauge group set to:  ",  Gauge_Group_String)
        
    elif Gauge_group == 3:
        os.environ["GAUGE_GROUP"]  = "su3"
        Gauge_Group_String         = "SU3"
        print("\nGauge group set to:",  Gauge_Group_String)
    
    else:
        raise ValueError("\nInvalid gauge group selection: Use 1 (SU(2) Complex),   2 (SU(2)),   3 (SU(3))")






    # Set precision mode of variables
    if Precision_mode == 1:
        os.environ["PRECISION"]  =  "single"                                                     # Single precision        
        Precision_Mode_String    =  "Single"
        print("Precision set to:    ", Precision_Mode_String, "(float32) \n")

    elif Precision_mode == 2:
        os.environ["PRECISION"]  =  "double"                                                    # Double precision      
        Precision_Mode_String    =  "Double"

        print("Precision set to:    ", Precision_Mode_String, "(float64) \n")                                                   

    else:
        raise ValueError("\nInvalid precision selection: Use 1 (Single) or 2 (Double) \n")

    return environment_vars, Gauge_Group_String

