import numpy as np



class Output_Quantities_CLASS():
    def __init__(self, glasma_params):

        self.glasma_params = glasma_params
        self.N_events = glasma_params['N_Events']
        self.N_times  = glasma_params['N_total_time_steps'+1]
        
        # Initialize storage for all quantities
        self.reset_storage()



        
    def reset_storage(self):
        """Initialize storage arrays"""
        self.Epsilon_0 = np.zeros((self.N_events, self.N_times))
        self.Pressure_L = np.zeros((self.N_events, self.N_times))
        self.Pressure_T = np.zeros((self.N_events, self.N_times))        
        self.taus = np.zeros(self.N_times)


    def store_event_data(self, i_event, i_time, Energy_density, Energy_density_prime, ly):           
        
        # Store main quantities
        self.Epsilon_0[i_event, i_time] = Energy_density.Epsilon_0 
        self.Pressure_L[i_event, i_time] = Energy_density.Pressure_L
        self.Pressure_T[i_event, i_time] = Energy_density.Pressure_T    
                
        # Store time once (same for all events)
        if i_event == 1:
            self.taus[i_time] = i_time * self.params['dt']
    

    def compute_averages(self):
        
        #Compute averages and statistical quantities across all events
        
        results = {
            
            'tau': self.taus,           # Time array
            
            # Mean values
            'Epsilon_0': np.mean(self.Epsilon_0, axis=0),
            'Pressure_L': np.mean(self.Pressure_L, axis=0),
            'Pressure_T': np.mean(self.Pressure_T, axis=0),
            
         
        }
        
        return results
    






    
    def write_time_step(self, i_time, i_event):
        """Write current time step data to files"""
        t = self.time[i_time]
        self.files['E_x'].write(f"{t} {self.E_x[i_time, i_event]}\n")
        self.files['E_y'].write(f"{t} {self.E_y[i_time, i_event]}\n")
        self.files['E_z'].write(f"{t} {self.E_z[i_time, i_event]}\n")
        self.files['B_x'].write(f"{t} {self.B_x[i_time, i_event]}\n")
        self.files['B_y'].write(f"{t} {self.B_y[i_time, i_event]}\n")
        self.files['B_z'].write(f"{t} {self.B_z[i_time, i_event]}\n")
        self.files['Epsilon_0'].write(f"{t} {self.Epsilon_0[i_time, i_event]}\n")
        self.files['Pressure_T'].write(f"{t} {self.Pressure_T[i_time, i_event]}\n")
        self.files['Pressure_L'].write(f"{t} {self.Pressure_L[i_time, i_event]}\n")
    
    def compute_averages(self):
        """Compute averages over all events"""
        print("\nComputing averages over all events...")
        print(f"Started at: {datetime.now()}")
        
        # Initialize average arrays
        self.av_E_x = np.zeros(self.N_time_steps+1)
        self.av_E_y = np.zeros(self.N_time_steps+1)
        self.av_E_z = np.zeros(self.N_time_steps+1)
        self.av_B_x = np.zeros(self.N_time_steps+1)
        self.av_B_y = np.zeros(self.N_time_steps+1)
        self.av_B_z = np.zeros(self.N_time_steps+1)
        
        self.av_E_T = np.zeros(self.N_time_steps+1)
        self.av_E_L = np.zeros(self.N_time_steps+1)
        self.av_B_T = np.zeros(self.N_time_steps+1)
        self.av_B_L = np.zeros(self.N_time_steps+1)
        self.av_Epsilon = np.zeros(self.N_time_steps+1)
        self.av_Pressure_T = np.zeros(self.N_time_steps+1)
        self.av_Pressure_L = np.zeros(self.N_time_steps+1)
        
        self.Ratio_PT_Epsilon = np.zeros(self.N_time_steps+1)
        self.Ratio_PL_Epsilon = np.zeros(self.N_time_steps+1)
        
        # Compute averages
        for i_time in range(self.N_time_steps+1):
            self.av_E_x[i_time] = np.mean(self.E_x[i_time, :])
            self.av_E_y[i_time] = np.mean(self.E_y[i_time, :])
            self.av_E_z[i_time] = np.mean(self.E_z[i_time, :])
            self.av_B_x[i_time] = np.mean(self.B_x[i_time, :])
            self.av_B_y[i_time] = np.mean(self.B_y[i_time, :])
            self.av_B_z[i_time] = np.mean(self.B_z[i_time, :])
            
            self.av_E_T[i_time] = np.mean(self.E_T[i_time, :])
            self.av_E_L[i_time] = np.mean(self.E_L[i_time, :])
            self.av_B_T[i_time] = np.mean(self.B_T[i_time, :])
            self.av_B_L[i_time] = np.mean(self.B_L[i_time, :])
            self.av_Epsilon[i_time] = np.mean(self.Epsilon_0[i_time, :])
            self.av_Pressure_T[i_time] = np.mean(self.Pressure_T[i_time, :])
            self.av_Pressure_L[i_time] = np.mean(self.Pressure_L[i_time, :])
            
            self.Ratio_PT_Epsilon[i_time] = self.av_Pressure_T[i_time] / self.av_Epsilon[i_time]
            self.Ratio_PL_Epsilon[i_time] = self.av_Pressure_L[i_time] / self.av_Epsilon[i_time]
        
        # Write average results to files
        self.write_averages()
        
        print(f"Finished at: {datetime.now()}")
    
    def write_averages(self):
        """Write averaged quantities to files"""
        with open('av_E_x.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_E_x[i_time]}\n")
        
        with open('av_E_y.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_E_y[i_time]}\n")
        
        with open('av_E_z.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_E_z[i_time]}\n")
        
        with open('av_B_x.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_B_x[i_time]}\n")
        
        with open('av_B_y.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_B_y[i_time]}\n")
        
        with open('av_B_z.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_B_z[i_time]}\n")
        
        with open('av_Epsilon_0.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_Epsilon[i_time]}\n")
        
        with open('av_Pressure_T.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_Pressure_T[i_time]}\n")
        
        with open('av_Pressure_L.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.av_Pressure_L[i_time]}\n")
        
        with open('Ratio_PT_Epsilon.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.Ratio_PT_Epsilon[i_time]}\n")
        
        with open('Ratio_PL_Epsilon.dat', 'w') as f:
            for i_time in range(self.N_time_steps+1):
                f.write(f"{self.time[i_time]} {self.Ratio_PL_Epsilon[i_time]}\n")
    
    def close_files(self):
        """Close all open files"""
        for f in self.files.values():
            f.close()

            ===================================================


class Output_Quantities_CLASS():
    def __init__(self, glasma_params):

        self.glasma_params = glasma_params
        self.N_events = glasma_params['N_Events']
        self.N_times  = glasma_params['N_total_time_steps'+1]
        
        # Initialize storage for all quantities
        self.reset_storage()



        
    def reset_storage(self):
        """Initialize storage arrays"""
        self.Epsilon_0 = np.zeros((self.N_events, self.N_times))
        self.Pressure_L = np.zeros((self.N_events, self.N_times))
        self.Pressure_T = np.zeros((self.N_events, self.N_times))        
        self.taus = np.zeros(self.N_times)


    def store_event_data(self, i_event, i_time, Energy_density, Energy_density_prime, ly):           
        
        # Store main quantities
        self.Epsilon_0[i_event, i_time] = Energy_density.Epsilon_0 
        self.Pressure_L[i_event, i_time] = Energy_density.Pressure_L
        self.Pressure_T[i_event, i_time] = Energy_density.Pressure_T    
                
        # Store time once (same for all events)
        if i_event == 1:
            self.taus[i_time] = i_time * self.params['dt']
    

    def compute_averages(self):
        
        #Compute averages and statistical quantities across all events
        
        results = {
            
            'tau': self.taus,           # Time array
            
            # Mean values
            'Epsilon_0': np.mean(self.Epsilon_0, axis=0),
            'Pressure_L': np.mean(self.Pressure_L, axis=0),
            'Pressure_T': np.mean(self.Pressure_T, axis=0),
            
         
        }
        
        return results