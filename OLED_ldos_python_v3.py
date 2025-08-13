import numpy as np  
import matplotlib.pyplot as plt  

# Define the materials and their HOMO and LUMO values  
materials = {  
    "NPB": {"type": "HTM", "HOMO": 5.4, "LUMO": 2.3},  
    "PEDOT": {"type": "HIM", "HOMO": 5.0, "LUMO": 3.0},  
    "Spiro-TAD": {"type": "HTM", "HOMO": 5.2, "LUMO": 2.4},  
    "Alq3": {"type": "EM", "HOMO": 5.9, "LUMO": 3.0},  
    "PFO": {"type": "Emissive", "HOMO": 5.8, "LUMO": 2.2},  
    "FIrpic": {"type": "Emissive", "HOMO": 5.1, "LUMO": 3.1},  
    "MEH-PPV": {"type": "Emissive", "HOMO": 5.0, "LUMO": 2.7},  
    "BPhen": {"type": "ETM", "HOMO": 6.5, "LUMO": 3.0},  
}  

# Define a function to plot the HOMO and LUMO levels  
def plot_homo_lumo(materials):  
    # Prepare data  
    material_names = list(materials.keys())  
    homo_values = [materials[name]['HOMO'] for name in material_names]  
    lumo_values = [materials[name]['LUMO'] for name in material_names]  

    # Create a figure  
    plt.figure(figsize=(10, 6))  

    x = np.arange(len(material_names))  # X positions for each material  

    # Plotting HOMO levels  
    plt.plot(x, homo_values, marker='o', label='HOMO', color='blue')  
    
    # Plotting LUMO levels  
    plt.plot(x, lumo_values, marker='o', label='LUMO', color='red')  

    # Add some labels and title  
    plt.title('HOMO and LUMO Levels of OLED Materials')  
    plt.xticks(x, material_names, rotation=45)  
    plt.axhline(0, color='black', lw=0.5, ls='--')  # Fermi energy level  
    plt.ylabel('Energy (eV)')  
    plt.ylim(0, 7)  # Set y-axis limit for better visualization  
    plt.grid()  
    plt.legend()  
    plt.tight_layout()  

    # Show plot  
    plt.show()  

# Call the function to plot  
plot_homo_lumo(materials)