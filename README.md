# Discovery_of_unsaturated_soil_model
The data and codes for the paper "Data-driven discovery of an explicit constitutive model for unsaturated hydraulic conductivity: Beyond the Mualem–van Genuchten model"

# Environments
python==3.12.4   
pandas==2.2.2  
numpy==1.26.4 
pytorch==2.3.1 
scikit-learn==1.4.2  
scipy==1.13.1 
matplotlib==3.8.4
sympy==1.13.3
openpyxl==3.1.5
torch-geometric==2.6.1


# The dataset utilized in this work 
* data for unsaturated flow (The dataset of water retention and unsaturated hydrualic conductivity that utilized in Van Genuchten, 1980), including:
  -Beit_Netofa_Clay  
  -Guelph Loam (drying)  
  -Guelph Loam (wetting)  
  -Hygiene sandstone  
  -Silt Loam G.E. 3  
  -Touchet Silt Loam G.E. 3  
* data from Unsoda dataset, inlucing 156 different soils.
  

# How to reproduce our work?
1. Install relevant packages (~20 min)  
2. Run the Equation discovery.py
   -model=='Train': Discover underlying equation from experimental data by graphic-symbolic regression, about 40 min for parallel computing and several hours for not parallel computing      
   Note: set self.use_parallel_computing=True for parallel computing.    
   -model=='Valid': Test the performance of the discovered equation  
   -model=='Valid discovered': Show the optimization process of graphic-symbolic regression with the top-5 equations in each epoch.  
3. Run the Valid_on_unsoda.py to test the performance of the discovered equation on Unsoda dataset (1~5 min).


# Expected outputs
* All outputs will be saved in the dir result_save  
* The discovered equations and corresponding awards are saved in "max_var_2", including best_fitness.pkl, best_graph.pkl which records the best 5 PDE structures in each optimization epoch.
  (The results is provided in this repository)
