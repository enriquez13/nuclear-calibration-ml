# src/calibration.py - VERSIÓN CORREGIDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

class NuclearCalibration:
    def __init__(self):
        # Data organized by particle type
        self.data_groups = {
            '4He+197Au': {
            'channel': np.array([7216,7232,7253,7175,7172,7255]),
            'energy': np.array([ 27.15,27.069, 27.101,26.956,27.036,27.128 ]),           
            'color': 'black', 'marker': 'o', 'label': '4He+197Au','zorder':'5'
            },
            '6He+197Au': {
                'channel': np.array([4644, 4609, 4590, 4588, 4542, 4563, 4625, 4583]),
                'energy': np.array([17.704, 17.615, 17.534, 17.649, 17.488, 17.576, 17.679, 17.437]),
                'color': 'blue', 'marker': 'D', 'label': '6He+197Au','zorder':'5'
            },
            '3H+197Au': {
                'channel': np.array([2412, 2422, 2408]),
                'energy': np.array([9.029, 9.008, 9.023]),
                'color': 'green', 'marker': 's', 'label': '6H+197Au','zorder':'5'
            },
            '4He+51V': {
                'channel': np.array([7183, 7177, 7224]),
                'energy': np.array([27.15, 26.984, 27.073]),
                'color': 'orange', 'marker': '^', 'label': '4He+51V','zorder':'5'
            },
            '6He+51V': {
                'channel': np.array([4669, 4605, 4607, 4521, 4650]),
                'energy': np.array([17.76, 17.489, 17.591, 17.377, 17.681]),
                'color': 'red', 'marker': '*', 'label': '6He+51V','zorder':'5'
            },
            '7Li+197Au': {
                'channel': np.array([3231, 3126, 3068, 2971, 2927, 3079, 3084, 3031]),
                'energy': np.array([14, 13.874, 13.757, 13.922, 13.688, 13.82, 13.964, 13.498]),
                'color': 'yellow', 'marker': 'p', 'label': '7Li+197Au (excluido)','zorder':5,
            },
            '7Li+51V': {
                'channel': np.array([3400, 3305, 3190, 3163, 3037, 3109, 3257]),
                'energy': np.array([14.366, 14.071, 13.809, 14.182, 13.66, 13.946, 14.28]),
                'color': 'cyan', 'marker': 'p', 'label': '7Li+51V (excluido)','zorder':5
            }
        }
        
        self.model = None
        self.ideal_model = None
    
    def train_models(self): 
        """Train the calibration models"""
        channels, energies = self.prepare_training_data()
        
        X = channels.reshape(-1, 1)
        y = energies
        
        # Main model
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Ideal model (y=0, x=0)
        self.ideal_model = LinearRegression(fit_intercept=False)
        self.ideal_model.fit(X, y)
        
        print("=== TRAINED MODELS ===")
        print(f"Model with regression: y(Energy)=mx(Channel)+b => Energy = {self.model.coef_[0]:.4f} × Channel + {self.model.intercept_:.4f}")
        print(f"Ideal model (b=0)=> Energy = {self.ideal_model.coef_[0]:.4f} × Channel")
    
    def prepare_training_data(self):  
        """Prepare data for training excluding 7Li"""
        channels = []
        energies = []
        
        exclude = ['7Li+197Au', '7Li+51V']
        
        for key in self.data_groups:
            if key not in exclude:
                group = self.data_groups[key]
                channels.extend(group['channel'])
                energies.extend(group['energy'])
        
        return np.array(channels), np.array(energies)
    
    def predict_li7(self):  
        """Prediction of where Li7 peaks should be"""
        li7_channels = np.concatenate([
            self.data_groups['7Li+197Au']['channel'],
            self.data_groups['7Li+51V']['channel']
        ])
        return self.model.predict(li7_channels.reshape(-1, 1))
    
    def analyze_results(self):  
        """Analyze and display the results"""
        li7_energies_real = np.concatenate([
            self.data_groups['7Li+197Au']['energy'],
            self.data_groups['7Li+51V']['energy']
        ])
        
        li7_energies_pred = self.predict_li7()
        errors = li7_energies_real - li7_energies_pred

            # Métricas principales para calibración
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / li7_energies_real)) * 100
        print("\n=== EVALUACIÓN DEL MODELO ===")
        print(f"📍 MAE  (Error Absoluto Medio): {mae:.4f} MeV")
        print(f"📊 MAPE (Error Porcentual):     {mape:.2f}%")  
        
        if len(errors) > 0:
            print(f"📈 Error máximo:              {np.max(np.abs(errors)):.4f} MeV")
            print(f"🎯 Discrepancia promedio:     {np.mean(errors):.4f} MeV")

    
    def plot_separate_graphs(self, save_figures=False):  # ✅ NUEVO MÉTODO
        """Genera DOS gráficas SEPARADAS"""
        # Gráfica 1: Calibración principal
        self._plot_main_calibration(save_figures)
        
        # Gráfica 2: Análisis Li7
        self._plot_li7_analysis(save_figures)
    
    def _plot_main_calibration(self, save_figures=False):  # ✅ CORREGIDO
        """Gráfica principal de calibración - SEPARADA"""
        plt.figure(figsize=(10, 8))
        
        # Configurar ejes
        plt.xlabel('Canal', fontsize=12)
        plt.ylabel('Energía (MeV)', fontsize=12)
        plt.title('Calibración de Canal para Energía', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Encontrar rango para las rectas
        all_channels = []
        for group in self.data_groups.values():
            all_channels.extend(group['channel'])
        
        x_range = np.linspace(0, max(all_channels) * 1.1, 100)
        
        # Dibujar rectas de calibración
        y_model = self.model.predict(x_range.reshape(-1, 1))
        y_ideal = self.ideal_model.predict(x_range.reshape(-1, 1))
        
        plt.plot(x_range, y_model, 'red', linewidth=2, label='Regresión lineal', zorder=1)
        plt.plot(x_range, y_ideal, 'blue', linestyle='--', linewidth=2, label='Recta ideal (0,0)', zorder=1)
        
        # Dibujar puntos de datos
        exclude = ['7Li+197Au', '7Li+51V']
        
        for key, group in self.data_groups.items():
            if key not in exclude:
                # Datos confiables
                plt.scatter(group['channel'], group['energy'],
                          c=group['color'], marker=group['marker'], s=80,
                          alpha=0.8, label=group['label'])
            else:
                # Datos excluidos (transparentes)
                plt.scatter(group['channel'], group['energy'],
                          c=group['color'], marker=group['marker'], s=80,
                          alpha=0.3, label=group['label'])
        
        # Configurar límites y leyenda
        plt.xlim(0, max(all_channels) * 1.1)
        plt.ylim(0, 30  * 1.1)
        plt.legend(loc='upper left', frameon=True, fancybox=True, 
                 shadow=True, framealpha=0.9, fontsize=15)
        
        plt.tight_layout()
        
        if save_figures:
            os.makedirs('../figures', exist_ok=True)
            plt.savefig('../figures/main_calibration.png', dpi=300, bbox_inches='tight')
            print("✅ Gráfica principal guardada como 'main_calibration.png'")
        
        plt.show()
    
    def _plot_li7_analysis(self, save_figures=False):  # ✅ CORREGIDO
        """Gráfica de análisis para 7Li"""
        plt.figure(figsize=(10, 8))
        
        li7_channels = np.concatenate([
            self.data_groups['7Li+197Au']['channel'],
            self.data_groups['7Li+51V']['channel']
        ])
        li7_energies_real = np.concatenate([
            self.data_groups['7Li+197Au']['energy'],
            self.data_groups['7Li+51V']['energy']
        ])
        li7_energies_pred = self.predict_li7()
        
        # Separar los dos conjuntos
        n_197au = len(self.data_groups['7Li+197Au']['channel'])
        
        # ⁷Li+¹⁹⁷Au
        plt.scatter(li7_channels[:n_197au], li7_energies_real[:n_197au],
                  c='yellow', marker='p', s=100, alpha=0.8,
                  label='7Li+197Au (Experimental)')
        plt.scatter(li7_channels[:n_197au], li7_energies_pred[:n_197au],
                  c='red', marker='x', s=150, linewidth=3,
                  label='7Li+197Au (Predicho)')
        
        # ⁷Li+⁵¹V
        plt.scatter(li7_channels[n_197au:], li7_energies_real[n_197au:],
                  c='cyan', marker='p', s=100, alpha=0.8,
                  label='7Li+51V (Experimental)')
        plt.scatter(li7_channels[n_197au:], li7_energies_pred[n_197au:],
                  c='blue', marker='x', s=150, linewidth=3,
                  label='7Li+51V (Predicho)')
        
        # Líneas de error
        for i in range(len(li7_channels)):
            plt.plot([li7_channels[i], li7_channels[i]],
                    [li7_energies_real[i], li7_energies_pred[i]],
                    'gray', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.xlabel('Canal', fontsize=12)
        plt.ylabel('Energía (MeV)', fontsize=12)
        plt.title('Predicciones vs Experimental - 7Li', fontsize=14)
        plt.legend(loc='upper left', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_figures:
            plt.savefig('../figures/li7_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Gráfica de Li7 guardada como 'li7_analysis.png'")
        
        plt.show()