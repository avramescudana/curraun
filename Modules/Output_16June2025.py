import numpy as np

class GlasmaOutput_CLASS:
    def __init__(self, N_events, N_time_steps):
        self.N_events = N_events
        self.N_time_steps = N_time_steps

        self.output_quantities       = [ "Epsilon_0", "Pressure_L", "Pressure_T", "ED_EL", "ED_ET", "ED_BL", "ED_BT" ]
        self.output_prime_quantities = [ "Epsilon_0_prime", "Pressure_L_prime", "Pressure_T_prime", "ED_EL_prime", "ED_ET_prime", "ED_BL_prime", "ED_BT_prime" ]        
        self.output_Lyapunov         = [ "EL_Ratio_Diff", "EL_Ratio_Diff_alpha2" ]
        

        # Combine all quantity names
        self.all_output_quantities = ( self.output_quantities + self.output_prime_quantities + self.output_Lyapunov )


        # Initialize storage arrays for all quantities with zeroes
        for i_output_quantity in self.all_output_quantities:
            setattr(self, i_output_quantity, np.zeros((N_events, N_time_steps + 1))) 

        # Time array
        self.time = np.zeros(N_time_steps+1)
        

        """
        # Initialize storage for statistics (using dictionaries)
        self.Average = {}
        self.Variance = {}
        self.StandardDeviation = {}
        self.Error = {}
        """

    # ======================================================================================================================================

       
    # ======================================================================================================================================   
    def store_event_data(self, i_Event, i_Time, Energy_density, Energy_density_prime, Ly):
        """Store data for a single event and time step"""
        
        for i_output_quantity in self.output_quantities:
            if hasattr(Energy_density, i_output_quantity):
                getattr(self, i_output_quantity)[i_Event, i_Time] = np.mean(getattr(Energy_density, i_output_quantity))

        for i_output_quantity in self.output_prime_quantities:
            if hasattr(Energy_density_prime, i_output_quantity):
                getattr(self, i_output_quantity)[i_Event, i_Time] = np.mean(getattr(Energy_density_prime, i_output_quantity))

        for i_output_quantity in self.output_Lyapunov:
            if hasattr(Ly, i_output_quantity):
                getattr(self, i_output_quantity)[i_Event, i_Time] = np.mean(getattr(Ly, i_output_quantity))
    # ======================================================================================================================================


    # ======================================================================================================================================
    """
    def compute_output_Avrg_Var_SD_Error(self):

        #Compute statistics across all events
        for i_output_quantity in self.all_output_quantities:
            data = getattr(self, i_output_quantity)
            self.Average[i_output_quantity]            =  np.mean(data, axis=0)                         # Compute Average
            self.Variance[i_output_quantity]           =  np.var(data, axis=0)                          # Compute Variance
            self.StandardDeviation[i_output_quantity]  =  np.std(data, axis=0)                          # Compute Standard Deviation
            self.Error[i_output_quantity]              =  np.std(data, axis=0)/np.sqrt(self.N_events)   # Compute Error

        output_dictionary = {'Average': self.Average,
                                'Variance': self.Variance,
                                'StandardDeviation': self.StandardDeviation,
                                'Error': self.Error}

        return output_dictionary    #self.Average, self.Variance, self.StandardDeviation, self.Error    
        """
    # ======================================================================================================================================

    def compute_output_Avrg_Var_SD_Error(self):
        """Compute statistics (Average, Variance, Standard Deviation, Error) across all events."""
        output_dict = {
            'Average': {},
            'Variance': {},
            'StandardDeviation': {},
            'Error': {}
        }

        for i_output_quantity in self.all_output_quantities:
            data = getattr(self, i_output_quantity)
            output_dict['Average'][i_output_quantity] = np.mean(data, axis=0)
            output_dict['Variance'][i_output_quantity] = np.var(data, axis=0)
            output_dict['StandardDeviation'][i_output_quantity] = np.std(data, axis=0)
            output_dict['Error'][i_output_quantity] = np.std(data, axis=0) / np.sqrt(self.N_events)


        """
        N_total_time_steps = 16
        dt  =  0.0625
        for i_time in range(N_total_time_steps + 1):
            time = i_time * dt
            
            Epsilon_0_Average = output_dict['Average']['Epsilon_0'][i_time]
            Epsilon_0_Error   = output_dict['Error']['Epsilon_0'][i_time]

            print(f"{time:.4f} \t {Epsilon_0_Average:.4e} ± {Epsilon_0_Error:.4e}")
        """

        
        return output_dict