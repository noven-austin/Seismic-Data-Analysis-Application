#Libraries used
import obspy                                                         # Seismic data processing
import pandas as pd                                                  # Handling CSV output
import tkinter as tk                                                 # GUI Components
from tkinter import ttk, filedialog, messagebox                      
import matplotlib.pyplot as plt                                      # Visualization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg      
from PyEMD import EMD                                                # Empirical Mode Decomposition of seismic signal
from tkinter import StringVar                                        
from scipy.signal import wiener                                      # Wiener filtering (noise reduction)
from obspy.taup import TauPyModel                                    # Calculating seismic wave travel times

# Earth Models used for time travel calculations
models = {
    "iasp91": TauPyModel(model="iasp91"),
    "ak135": TauPyModel(model="ak135"),
    "prem": TauPyModel(model="prem")
}

class SeismicApp:
    def __init__(self, root):
        # Initialize main window, title, and size.
        self.root = root
        self.root.title("Seismic Time Series Analyzer")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Top frame contains button for upload and options.
        self.top_frame = ttk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Upload SAC file button.
        self.upload_btn = ttk.Button(self.top_frame, text="Upload SAC File", command=self.upload_file)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Dropdown menu to select earth model.
        self.model_var = StringVar(value="iasp91")
        self.model_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.model_var, values=list(models.keys()), state="readonly", width=10
        )
        self.model_dropdown.pack(side=tk.LEFT, padx=10)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_model_selection)

        # Checkbox for weiner filter application on seismic data.
        self.wiener_value = tk.IntVar(value=0)
        self.wiener_check = ttk.Checkbutton(self.top_frame, variable=self.wiener_value, text="Wiener", command=self.toggle_wiener)
        self.wiener_check.pack(side=tk.LEFT, padx= 5)

        # Frame to contain metadata and arrival time information.
        self.info_frame = ttk.Frame(root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Frame to display metadata extracted from the SAC file.
        self.metadata_frame = ttk.LabelFrame(self.info_frame, text="Metadata", padding=(5, 5), relief="solid", borderwidth=2)
        self.metadata_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.metadata_label = ttk.Label(self.metadata_frame, text="Metadata will appear here", justify="left", anchor="w")
        self.metadata_label.pack(anchor="nw", padx=5, pady=5)

        # Frame to display calculated seismis phase arrival time.
        self.arrival_frame = ttk.LabelFrame(self.info_frame, text="Arrival Times", padding=(5, 5), relief="solid", borderwidth=2)
        self.arrival_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 
        self.arrival_content_frame = ttk.Frame(self.arrival_frame)
        self.arrival_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        #
        self.arrival_times_top = ttk.LabelFrame(self.arrival_content_frame, text="Current Arrival Times", padding=(5, 5), relief="solid", borderwidth=2)
        self.arrival_times_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.arrival_label = ttk.Label(self.arrival_times_top, text="Arrival Times will appear here", justify="left", anchor="w")
        self.arrival_label.pack(anchor="nw", padx=5, pady=5)

        # Lower subframe containing additional functions ( Seismic phase splitting using EMD )
        self.arrival_times_bottom = ttk.LabelFrame(self.arrival_content_frame, text="Additional Functions", padding=(5, 5), relief="solid", borderwidth=2)
        self.arrival_times_bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Button to process and split P-wave using EMD.
        self.split_p_btn = ttk.Button(self.arrival_times_bottom, text="Split P", command=lambda: self.applyEMD("p"),state=tk.DISABLED)
        self.split_p_btn.pack(side=tk.LEFT, padx=5, pady=5)
        # Button to process and split S-wave using EMD.
        self.split_s_btn = ttk.Button(self.arrival_times_bottom, text="Split S", command=lambda: self.applyEMD("s"), state=tk.DISABLED)
        self.split_s_btn.pack(side=tk.LEFT, padx=5, pady=5)
        # Button to revert to the original signal.
        self.revert_btn = ttk.Button(self.arrival_times_bottom, text="View Original", command=self.revert_to_original, state=tk.DISABLED)
        self.revert_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame containing the download button for CSV data and image.
        self.download_frame = ttk.LabelFrame(self.info_frame, text="Download Options", padding=(5, 5), relief="solid", borderwidth=2)
        self.download_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.download_data_btn = ttk.Button(self.download_frame, text="Download .csv", command=self.download_data, state=tk.DISABLED)
        self.download_data_btn.pack(pady=5)
        self.download_image_btn = ttk.Button(self.download_frame, text="Download Figure", command=self.download_image, state=tk.DISABLED)
        self.download_image_btn.pack(pady=5)

        # Frame to display the seismic time series.
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Variable initializations
        self.file_data = None
        self.arrival_times = {}
        self.selected_model = "iasp91"

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("SAC files", "*.sac")])
        if not file_path:
            return

        try:
            st = obspy.read(file_path)
            self.file_data = st[0]
            self.original_times = self.file_data.times()
            self.original_data = self.file_data.data
            self.display_metadata()
            self.calculate_arrival_times()
            self.plot_time_series()
            self.download_data_btn.config(state=tk.NORMAL)
            self.download_image_btn.config(state=tk.NORMAL)
            self.split_p_btn.config(state=tk.NORMAL)
            self.split_s_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read SAC file: {e}")

    def toggle_wiener(self):
        if not self.file_data:
            messagebox.showerror("Error", "No data loaded to apply Wiener Filter")
            self.wiener_value.set(0)
            return
        
        if self.wiener_value.get():
            self.filtered_data = wiener(self.original_data, mysize=20)
            self.file_data.data = self.filtered_data
        else:
            self.file_data.data = self.original_data
            self.filtered_data = None
        
        self.calculate_arrival_times()
        self.plot_time_series()

    def display_metadata(self):
        stats = self.file_data.stats
        metadata = f"""
        Network Code: {stats.sac.knetwk}
        Station: {stats.station}
        Channel: {stats.sac.kcmpnm}
        Start Time: {stats.starttime.strftime('%Y-%m-%d %H:%M:%S')}
        End Time: {stats.endtime.strftime('%Y-%m-%d %H:%M:%S')}
        Sampling Rate: {stats.sampling_rate} Hz
        Station Latitude: {stats.sac.stla}
        Station Longitude: {stats.sac.stlo}
        Depth: {stats.sac.evdp}
        Distance: {stats.sac.gcarc}
        """
        self.metadata_label.config(text=metadata)
        self.Station = stats.station
        self.Channel = stats.sac.kcmpnm

    def revert_to_original(self):
        if not hasattr(self, 'original_times') or not hasattr(self, 'original_data'):
            messagebox.showerror("Error", "Original data is not available.")
            return

        self.ax.clear()
        self.ax.plot(self.original_times, self.original_data, label="Original Time Series")

        p_wave_time = self.arrival_times.get(self.selected_model, {}).get("P", None)
        s_wave_time = self.arrival_times.get(self.selected_model, {}).get("S", None)

        if p_wave_time:
            self.ax.axvline(p_wave_time, color='red', linestyle='--', label=f"{self.selected_model} P-Wave")
        if s_wave_time:
            self.ax.axvline(s_wave_time, color='blue', linestyle='--', label=f"{self.selected_model} S-Wave")

        self.ax.set_title(f"Seismic Time Series of Station {self.Station}, Channel {self.Channel} with Arrival Times")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Count")
        self.ax.legend()
        self.canvas.draw()
        self.revert_btn.config(state=tk.DISABLED)


    def applyEMD(self, wave_type):
        if not self.arrival_times or not self.file_data:
            messagebox.showerror("Error", "No arrival times or data available for processing.")
            return
        
        self.revert_btn.config(state=tk.NORMAL)
        sampling_rate = self.file_data.stats.sampling_rate

        max_time = 600
        end_index = min(int(max_time * sampling_rate), len(self.file_data.data))
        trimmed_data = self.file_data.data[:end_index]
        trimmed_times = self.file_data.times()[:end_index]

        p_wave_time = self.arrival_times.get(self.selected_model, {}).get("P", None)
        s_wave_time = self.arrival_times.get(self.selected_model, {}).get("S", None)

        seismicEmd = EMD()
        seisIMFS = seismicEmd(trimmed_data)

        if len(seisIMFS) < 7:
            messagebox.showerror("Error", f"Insufficient IMFs generated ({len(seisIMFS)} found). At least 7 are required.")
            return

        self.ax.clear()
        if wave_type == "p":
            self.reconstructed_signal = seisIMFS[1] + seisIMFS[6] + seisIMFS[7]
            self.ax.plot(trimmed_times, trimmed_data, label="Original Signal", alpha=0.5)
            self.ax.plot(trimmed_times, self.reconstructed_signal, label="Reconstructed Signal (P-wave)")
            plot_title = f"P-Wave Reconstruction\nArrival Time: {p_wave_time:.2f} seconds" if p_wave_time else "P-Wave Reconstruction"
        else:
            self.reconstructed_signal = seisIMFS[2] + seisIMFS[3] + seisIMFS[4] + seisIMFS[5]
            self.ax.plot(trimmed_times, trimmed_data, label="Original Signal", alpha=0.5)
            self.ax.plot(trimmed_times, self.reconstructed_signal, label="Reconstructed Signal (S-wave)")
            plot_title = f"S-Wave Reconstruction\nArrival Time: {s_wave_time:.2f} seconds" if s_wave_time else "S-Wave Reconstruction"

        if p_wave_time and p_wave_time <= max_time:
            self.ax.axvline(p_wave_time, color='red', linestyle='--', label="P-Wave Arrival")
        if s_wave_time and s_wave_time <= max_time:
            self.ax.axvline(s_wave_time, color='blue', linestyle='--', label="S-Wave Arrival")

        self.ax.set_title(plot_title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.canvas.draw()

    def calculate_arrival_times(self):
        self.arrival_times.clear()
        source_depth = self.file_data.stats.sac.evdp
        epicentral_distance = self.file_data.stats.sac.gcarc

        if source_depth is None or epicentral_distance is None:
            messagebox.showerror("Error", "SAC file is missing required metadata (evdp or dist).")
            return

        selected_model = self.selected_model
        arrival_text = f"Arrival Times for {selected_model} (in seconds):\n"
        try:
            arrivals = models[selected_model].get_travel_times(
                source_depth_in_km=source_depth, distance_in_degree=epicentral_distance, phase_list=["P", "S"]
            )
            arrivalDict = {arrival.name: arrival.time for arrival in arrivals}
            self.arrival_times[selected_model] = {
                "P": arrivalDict.get("P", None),
                "S": arrivalDict.get("S", None),
            }

            p_wave_time = f"{self.arrival_times[selected_model]['P']:.2f}s" if self.arrival_times[selected_model]['P'] else "Not available"
            s_wave_time = f"{self.arrival_times[selected_model]['S']:.2f}s" if self.arrival_times[selected_model]['S'] else "Not available"
            arrival_text += f"P-Wave: {p_wave_time}\nS-Wave: {s_wave_time}"

        except Exception as e:
            arrival_text += f"Error: {str(e)}"

        self.arrival_label.config(text=arrival_text)

    def plot_time_series(self):
        self.ax.clear()
        times = self.file_data.times()
        data = self.file_data.data
        self.ax.plot(times, data)

        selected_model = self.selected_model
        p_wave_time = self.arrival_times.get(selected_model, {}).get("P", None)
        s_wave_time = self.arrival_times.get(selected_model, {}).get("S", None)

        if p_wave_time:
            self.ax.axvline(p_wave_time, color='red', linestyle='--', label=f"{selected_model} P-Wave")
        if s_wave_time:
            self.ax.axvline(s_wave_time, color='blue', linestyle='--', label=f"{selected_model} S-Wave")

        self.ax.set_title(f"Seismic Time Series of Station {self.Station}, Channel {self.Channel} with Arrival Times")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Count")
        self.ax.legend()
        self.canvas.draw()

    def update_model_selection(self, event):
        self.selected_model = self.model_var.get()
        if self.file_data:
            self.calculate_arrival_times()
            self.plot_time_series()

    def download_data(self):
        if self.file_data is None:
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        df = pd.DataFrame({
            "Time (s)": self.file_data.times(),
            "Count": self.file_data.data
        })
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", "Data saved successfully!")

    def download_image(self):
        if self.file_data is None:
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return

        self.figure.savefig(file_path)
        messagebox.showinfo("Success", "Plot image saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SeismicApp(root)
    root.mainloop()
