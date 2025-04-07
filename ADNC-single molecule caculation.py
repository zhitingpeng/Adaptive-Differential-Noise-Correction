import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

# Suppress RankWarning from polyfit
warnings.simplefilter('ignore', np.RankWarning)

# Global variables to store ds, ds_prime, ds_double_prime and adjusted values
recorded_ds = None
recorded_ds_prime = None
recorded_ds_double_prime = None
adjusted_ds_prime = None
adjusted_ds = None
adjusted_ds_double_prime = None

def count_bright_spots(image_path, threshold=5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bright_spots_count = len(contours)
    return bright_spots_count, binary_image, contours

def select_image(sample_index):
    file_path = filedialog.askopenfilename()
    if file_path:
        image_paths[sample_index] = file_path
        bright_spots_count, _, _ = count_bright_spots(file_path, threshold_slider.get())
        labels[sample_index].config(text=f"Bright spots count: {bright_spots_count}")
        update_images()

def delete_all_images():
    for i in range(len(image_paths)):
        image_paths[i] = None
        labels[i].config(text="No image selected")
        image_labels[i].config(image='')

def clear_recorded_ds():
    global recorded_ds, recorded_ds_prime, recorded_ds_double_prime
    recorded_ds = None
    recorded_ds_prime = None
    recorded_ds_double_prime = None
    ds_label.config(text="ds: ")
    ds_prime_label.config(text="ds': ")
    ds_double_prime_label.config(text="ds'': ")
    x_label.config(text="Recorded ds values cleared.")
    clear_plot()

def update_images(*args):
    for i, image_path in enumerate(image_paths):
        if image_path:
            bright_spots_count, binary_image, contours = count_bright_spots(image_path, threshold_slider.get())
            img = Image.fromarray(binary_image)
            img.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img)
            image_labels[i].config(image=img)
            image_labels[i].image = img
            labels[i].config(text=f"Bright spots count: {bright_spots_count}")
    calculate_and_display_concentration()

def get_signals():
    signals = []
    for image_path in image_paths:
        if image_path:
            bright_spots_count, _, _ = count_bright_spots(image_path, threshold_slider.get())
            signals.append(bright_spots_count)
    return signals

def update_images_with_best_threshold():
    try:
        Q = float(quality_sample_concentration.get())
        C = float(control_concentration.get())
    except ValueError:
        x_label.config(text="Please enter valid concentrations for Q and C before adjusting the threshold.")
        return

    initial_threshold = threshold_slider.get()
    best_threshold = initial_threshold
    
    min_diff = float('inf')

    for t in range(0, 256, 3):  # Adjust the step size as needed
        threshold_slider.set(t)
        update_images()
        
        signals = get_signals()
        if len(signals) < 6:
            x_label.config(text="Please select all required images.")
            return
        
        S_0, S_t1, S_c1, S_t2, S_c2, S_t3 = signals
        S_Q, S_c = S_c1, S_c2
        
        if check_threshold_conditions(S_c, S_Q, S_0, C, Q):
            delta = [1.0, 1.0, 1.0]
            x_avg, _, _, _, xmean = calculate_concentration(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, t)
            diff = abs(x_avg - xmean)
            
            if diff < min_diff:
                min_diff = diff
                best_threshold = t

    threshold_slider.set(best_threshold)
    update_images()

def check_threshold_conditions(S_c, S_Q, S_0, C, Q):
    ds = (S_c - S_Q) / (C - Q)
    ds_prime = (S_c - S_0) / C
    ds_double_prime = (S_Q - S_0) / Q
    return ds > ds_prime > ds_double_prime > 0

def loss_function(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold):
    xmean, x1, x2, x3, _ = calculate_concentration(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold)
    if np.isnan(xmean) or np.isnan(x1) or np.isnan(x2) or np.isnan(x3):
        return float('inf')
    return (x1 - x2) ** 2 + (x2 - x3) ** 2 + (x3 - x1) ** 2

def calculate_gradient(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold, epsilon=1e-5):
    grad_delta = np.zeros_like(delta, dtype=float)
    for i in range(len(delta)):
        delta_plus = np.copy(delta)
        delta_plus[i] += epsilon
        loss1 = loss_function(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold)
        loss2 = loss_function(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta_plus, threshold)
        grad_delta[i] = (loss2 - loss1) / epsilon
        if np.isnan(grad_delta[i]):
            grad_delta[i] = 0.0
    
    threshold_plus = threshold + epsilon
    loss1 = loss_function(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold)
    loss2 = loss_function(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold_plus)
    grad_threshold = (loss2 - loss1) / epsilon
    if np.isnan(grad_threshold):
        grad_threshold = 0.0
    
    return grad_delta, grad_threshold

def optimize_threshold_and_weights(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, initial_threshold, initial_delta, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
    threshold = initial_threshold
    delta = np.array(initial_delta, dtype=float)
    for _ in range(max_iterations):
        grad_delta, grad_threshold = calculate_gradient(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold)
        new_delta = delta - learning_rate * grad_delta
        new_threshold = threshold - learning_rate * grad_threshold
        if np.linalg.norm(new_delta - delta) < tolerance and abs(new_threshold - threshold) < tolerance:
            break
        delta = new_delta
        threshold = new_threshold
    return threshold, delta

def calculate_and_display_concentration():
    try:
        Q = float(quality_sample_concentration.get())
        C = float(control_concentration.get())
    except ValueError:
        x_label.config(text="Please enter valid concentrations for Q and C.")
        return
    
    signals = get_signals()
    if len(signals) < 6:
        x_label.config(text="Please select all required images.")
        return
    
    S_0, S_t1, S_c1, S_t2, S_c2, S_t3 = signals
    S_Q, S_c = S_c1, S_c2
    
    initial_delta = [1.0, 1.0, 1.0]
    initial_threshold = threshold_slider.get()
    
    threshold, optimized_delta = optimize_threshold_and_weights(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, initial_threshold, initial_delta)
    
    x_avg, x1, x2, x3, xmean = calculate_concentration(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, optimized_delta, threshold)
    
    if np.isnan(x_avg):
        x_label.config(text="Invalid test. Please check your inputs.")
    else:
        x_label.config(text=f"Concentration: {x_avg:.3f}, xmean: {xmean:.3f}")
        plot_standard_curve(Q, C, S_0, S_t1, S_t2, S_t3, S_Q, S_c)

def calculate_concentration(S_t1, S_t2, S_t3, S_0, S_Q, S_c, Q, C, delta, threshold):
    global recorded_ds, recorded_ds_prime, recorded_ds_double_prime
    global adjusted_ds_prime, adjusted_ds, adjusted_ds_double_prime

    # Calculate ds, ds_prime, ds_double_prime only if not already recorded
    if recorded_ds is None or recorded_ds_prime is None or recorded_ds_double_prime is None:
        ds = (S_c - S_Q) / (C - Q)
        ds_prime = (S_c - S_0) / C
        ds_double_prime = (S_Q - S_0) / Q

        if ds > ds_prime > ds_double_prime > 0:
            adjusted_ds_prime = delta[1] * ds_prime
            adjusted_ds = delta[0] * ds
            adjusted_ds_double_prime = delta[2] * ds_double_prime
            recorded_ds = adjusted_ds
            recorded_ds_prime = adjusted_ds_prime
            recorded_ds_double_prime = adjusted_ds_double_prime
            ds_label.config(text=f"ds: {ds:.3f}")
            ds_prime_label.config(text=f"ds': {ds_prime:.3f}")
            ds_double_prime_label.config(text=f"ds'': {ds_double_prime:.3f}")
        else:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    else:
        ds = recorded_ds
        ds_prime = recorded_ds_prime
        ds_double_prime = recorded_ds_double_prime

    if S_Q <= S_t1:
        x1 = Q + (S_t1 - S_Q) / ds
        x2 = C + (S_t2 - S_c) / ds
        denominator = (S_t1 - S_Q - S_t2 + S_c)
        if denominator == 0:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        x_avg = ((S_t1 - S_Q) * C + (S_c - S_t2) * Q) / denominator
        xmean = (x1 + x2) / 2
        return x_avg, x1, x2, float('nan'), xmean
    else:
        x_min = (S_t3 - S_0) / adjusted_ds_double_prime
        x_max = Q
        x_avg = (x_min + x_max) / 2
        x1 = Q + (S_t1 - S_Q) / adjusted_ds_prime
        x2 = C + (S_t2 - S_c) / adjusted_ds
        x3 = (S_t3 - S_0) / adjusted_ds_double_prime
        xmean = (x1 + x2 + x3) / 3
        return x_avg, x_min, x_max, x3, xmean
    
def calculate_new_concentration():
    try:
        Q = float(quality_sample_concentration.get())
        C = float(control_concentration.get())
    except ValueError:
        x_label.config(text="Please enter valid concentrations for Q and C.")
        return
    
    signals = get_signals()
    
    concentrations = []
    signals_list = []

    if adjusted_ds_prime is None or adjusted_ds is None or adjusted_ds_double_prime is None:
        x_label.config(text="Please ensure initial images are processed to record ds values.")
        return

    if image_paths[2] and image_paths[3] and len(signals) >= 2:  # Test sample 1 and Quantitation Sample
        S_Q, S_t1 = signals[:2]
        
        if S_Q <= S_t1:
              x1 = Q + (S_t1 - S_Q) / adjusted_ds
        else:
              x1 = Q + (S_t1 - S_Q) / adjusted_ds_double_prime
            
        concentrations.append(x1)
        signals_list.append(S_t1)
        
    if image_paths[0] and image_paths[1] and len(signals) >= 2:  # Test sample 3 and Blank Sample
        S_0, S_t3 = signals[:2]
        x3 = (S_t3 - S_0) / adjusted_ds_double_prime
        concentrations.append(x3)
        signals_list.append(S_t3)
        
    if image_paths[4] and image_paths[5] and len(signals) >= 2:  # Test sample 2 and Control Sample
        S_c, S_t2 = signals[:2]
        x2 = C + (S_t2 - S_c) / adjusted_ds
        concentrations.append(x2)
        signals_list.append(S_t2)
    
    if image_paths[2] and image_paths[3] and image_paths[4] and image_paths[5]:  # Test sample 2 and Control Sample
        S_Q, S_t1, S_c, S_t2 = signals[:4]
        x2 = ((S_t1 - S_Q) * C + (S_c - S_t2) * Q) / (S_t1 - S_Q - S_t2 + S_c)
        xmean = (x1 + x2) / 2
        concentrations.append(xmean)
        signals_list.append((S_t1 + S_t2) / 2)
    
    if image_paths[0] and image_paths[1] and image_paths[2] and image_paths[3]:  # Test sample 3 and Blank Sample
        S_0, S_t3, S_Q, S_t1 = signals[:4]
        x3 = ((S_t3 - S_0) / adjusted_ds_double_prime + Q) / 2
        xmean = (x1 + x3) / 2
        concentrations.append(xmean)
        signals_list.append((S_t1 + S_t3) / 2)

    if len(concentrations) > 0:
        x_avg = np.mean(concentrations)
        if x_avg < 0:
            x_label.config(text=f"Average Concentration: {x_avg:.3f} (Negative value detected, check signal inputs)\n" + "\n".join([f"x{i+1}: {concentrations[i]:.3f}" for i in range(len(concentrations))]))
        else:
            x_label.config(text=f"Average Concentration: {x_avg:.3f}\n" + "\n".join([f"x{i+1}: {concentrations[i]:.3f}" for i in range(len(concentrations))]))
            plot_new_points(concentrations, signals_list)

def signal_curve(concentration, coeffs):
    return np.polyval(coeffs, concentration)

def plot_standard_curve(Q, C, S_0, S_t1, S_t2, S_t3, S_Q, S_c):
    # Close all previous figures
    plt.close('all')
    
    fig, ax = plt.subplots()
    
    concentrations = [0, Q, C]
    signals = [S_0, S_Q, S_c]
    
    coeffs = np.polyfit(concentrations, signals, 2)
    
    fitted_concentrations = np.linspace(0, max(concentrations) * 1.5, 100)
    fitted_signals = signal_curve(fitted_concentrations, coeffs)
    
    ax.plot(fitted_concentrations, fitted_signals, label='Fitted Curve')
    
    ax.scatter(concentrations, signals, color='red', label='Known Points')
    
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Signal')
    ax.legend()
    
    clear_plot()
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)

def plot_new_points(concentrations, signals):
    plt.scatter(concentrations, signals, color='blue', label='Test Points')
    plt.draw()

def clear_plot():
    for widget in plot_frame.winfo_children():
        widget.destroy()

# GUI Setup
root = Tk()
root.title("Concentration Calculation")

mainframe = ttk.Frame(root, padding="10 10 20 20")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

image_paths = [None, None, None, None, None, None]
labels = []
image_labels = []

# Adjust the layout to place images on the left and controls on the right
image_frame = ttk.Frame(mainframe)
image_frame.grid(column=0, row=0, rowspan=6, sticky=(N, W, E, S))
control_frame = ttk.Frame(mainframe)
control_frame.grid(column=1, row=0, rowspan=6, sticky=(N, W, E, S))

# Adjust the layout of image_frame to be 3 rows and 2 columns
for i, sample_name in enumerate(["Blank sample", "Test sample 3", "Quantitation Sample", "Test sample 1", "Control sample", "Test sample 2"]):
    row = i // 2
    col = i % 2
    ttk.Label(image_frame, text=sample_name).grid(row=row*3, column=col*3, sticky=W)
    label = ttk.Label(image_frame, text="No image selected")
    label.grid(row=row*3+1, column=col*3, sticky=(W, E))
    labels.append(label)
    image_label = Label(image_frame)
    image_label.grid(row=row*3+2, column=col*3, columnspan=2, sticky=W)
    image_labels.append(image_label)
    ttk.Button(image_frame, text="Select Image", command=lambda i=i: select_image(i)).grid(row=row*3+1, column=col*3+2, sticky=E)

# Adding input fields for Quality Sample Concentration (Q) and Control Sample Concentration (C) in control_frame
ttk.Label(control_frame, text="Quantitation Sample Concentration (Q):").grid(row=0, column=0, sticky=W)
quality_sample_concentration = StringVar()
ttk.Entry(control_frame, textvariable=quality_sample_concentration).grid(row=0, column=1, columnspan=3, sticky=(W, E))

ttk.Label(control_frame, text="Control Sample Concentration (C):").grid(row=1, column=0, sticky=W)
control_concentration = StringVar()
ttk.Entry(control_frame, textvariable=control_concentration).grid(row=1, column=1, columnspan=3, sticky=(W, E))

# Adding threshold slider and button in control_frame
ttk.Label(control_frame, text="Threshold:").grid(row=2, column=0, sticky=W)
threshold_slider = Scale(control_frame, from_=0, to=255, orient=HORIZONTAL, command=update_images)
threshold_slider.set(5)
threshold_slider.grid(row=2, column=1, columnspan=2, sticky=(W, E))

ttk.Button(control_frame, text="Auto Adjust Threshold", command=update_images_with_best_threshold).grid(row=2, column=3, sticky=E)

# Labels to display ds, ds_prime, ds_double_prime in control_frame
ds_label = ttk.Label(control_frame, text="ds: ")
ds_label.grid(row=3, column=0, columnspan=4)
ds_prime_label = ttk.Label(control_frame, text="ds': ")
ds_prime_label.grid(row=4, column=0, columnspan=4)
ds_double_prime_label = ttk.Label(control_frame, text="ds'': ")
ds_double_prime_label.grid(row=5, column=0, columnspan=4)

# Label to display calculated concentration in control_frame
x_label = ttk.Label(control_frame, text="")
x_label.grid(row=6, column=0, columnspan=4)

# Button to delete all images in control_frame
ttk.Button(control_frame, text="Delete All Images", command=delete_all_images).grid(row=7, column=0, columnspan=4)

# Button to clear recorded ds values in control_frame
ttk.Button(control_frame, text="Clear Recorded ds Values", command=clear_recorded_ds).grid(row=8, column=0, columnspan=4)

# Button to calculate new concentration based on new images and recorded ds values in control_frame
ttk.Button(control_frame, text="Calculate New Concentration", command=calculate_new_concentration).grid(row=9, column=0, columnspan=4)

# Frame to display the plot
plot_frame = ttk.Frame(control_frame)
plot_frame.grid(row=10, column=0, columnspan=4, sticky=(N, S, E, W))

root.mainloop()
