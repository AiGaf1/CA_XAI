import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import plotly.graph_objects as go
from torch import nn
import numpy as np
import seaborn as sns
import math
from models.LTE import LearnableFourierFeatures

# KEYBOARD_LAYOUT = {
#     # Row 1 (numbers)
#     '`': (0, 0), '1': (1, 0), '2': (2, 0), '3': (3, 0), '4': (4, 0),
#     '5': (5, 0), '6': (6, 0), '7': (7, 0), '8': (8, 0), '9': (9, 0),
#     '0': (10, 0), '-': (11, 0), '=': (12, 0),
#
#     # Row 2 (QWERTY)
#     'q': (1.5, 1), 'w': (2.5, 1), 'e': (3.5, 1), 'r': (4.5, 1), 't': (5.5, 1),
#     'y': (6.5, 1), 'u': (7.5, 1), 'i': (8.5, 1), 'o': (9.5, 1), 'p': (10.5, 1),
#     '[': (11.5, 1), ']': (12.5, 1), '\\': (13.5, 1),
#
#     # Row 3 (ASDF)
#     'a': (1.75, 2), 's': (2.75, 2), 'd': (3.75, 2), 'f': (4.75, 2), 'g': (5.75, 2),
#     'h': (6.75, 2), 'j': (7.75, 2), 'k': (8.75, 2), 'l': (9.75, 2),
#     ';': (10.75, 2), "'": (11.75, 2),
#
#     # Row 4 (ZXCV)
#     'z': (2.25, 3), 'x': (3.25, 3), 'c': (4.25, 3), 'v': (5.25, 3), 'b': (6.25, 3),
#     'n': (7.25, 3), 'm': (8.25, 3), ',': (9.25, 3), '.': (10.25, 3), '/': (11.25, 3),
#
#     # Space bar
#     ' ': (7, 4),
# }
#
#
# def visualize_keystrokes(keystrokes_data):
#     """
#     Visualize keystroke data on a keyboard layout.
#
#     Parameters:
#     keystrokes_data: list of dicts with keys 'keycode', 'hold_time', 'flight_time'
#                      Example: [{'keycode': 97, 'hold_time': 0.15, 'flight_time': 0.12}, ...]
#     """
#     fig, ax = plt.subplots(figsize=(16, 8))
#
#     # Draw all keyboard keys in light gray
#     for key, (x, y) in KEYBOARD_LAYOUT.items():
#         width = 2 if key == ' ' else 0.9
#         rect = patches.Rectangle((x, -y), width, 0.9,
#                                  linewidth=1, edgecolor='gray',
#                                  facecolor='lightgray', alpha=0.3)
#         ax.add_patch(rect)
#         ax.text(x + width / 2, -y + 0.45, key,
#                 ha='center', va='center', fontsize=10, color='gray')
#
#     # Group keystrokes by character to handle repeated letters
#     char_occurrences = {}
#     for idx, keystroke in enumerate(keystrokes_data):
#         char = chr(keystroke['keycode']).lower()
#         if char not in char_occurrences:
#             char_occurrences[char] = []
#         char_occurrences[char].append({
#             'index': idx,
#             'hold_time': keystroke['hold_time'],
#             'flight_time': keystroke['flight_time']
#         })
#
#     # Draw pressed keys with hold time information
#     for char, occurrences in char_occurrences.items():
#         if char in KEYBOARD_LAYOUT:
#             x, y = KEYBOARD_LAYOUT[char]
#             width = 2 if char == ' ' else 0.9
#
#             # Calculate positions for multiple order numbers
#             num_occurrences = len(occurrences)
#
#             # Draw order numbers in a row
#             if num_occurrences == 1:
#                 # Single occurrence - center position
#                 positions = [(x + 0.15, -y + 0.15)]
#             elif num_occurrences == 2:
#                 # Two occurrences - left and center
#                 positions = [(x + 0.15, -y + 0.15), (x + width / 2, -y + 0.15)]
#             elif num_occurrences == 3:
#                 # Three occurrences - left, center, right on top row
#                 positions = [(x + 0.15, -y + 0.15), (x + width / 2, -y + 0.15), (x + width - 0.15, -y + 0.15)]
#             else:
#                 # Four or more - arrange in two rows
#                 top_row = (num_occurrences + 1) // 2
#                 positions = []
#                 for i in range(top_row):
#                     positions.append((x + 0.15 + i * (width - 0.3) / (top_row - 1 if top_row > 1 else 1), -y + 0.12))
#                 for i in range(num_occurrences - top_row):
#                     bottom_count = num_occurrences - top_row
#                     positions.append(
#                         (x + 0.15 + i * (width - 0.3) / (bottom_count - 1 if bottom_count > 1 else 1), -y + 0.35))
#
#             # Draw each occurrence
#             for occurrence, pos in zip(occurrences, positions):
#                 circle = patches.Circle(pos, 0.08,
#                                         facecolor='white', edgecolor='black', linewidth=1.5)
#                 ax.add_patch(circle)
#                 ax.text(pos[0], pos[1], str(occurrence['index'] + 1),
#                         ha='center', va='center', fontsize=8, fontweight='bold')
#
#             # Add key character
#             ax.text(x + width / 2, -y + 0.45, char,
#                     ha='center', va='center', fontsize=14, fontweight='bold')
#
#             # Add hold times below key (show all occurrences)
#             hold_times_text = ', '.join([f"{occ['hold_time']:.3f}s" for occ in occurrences])
#             ax.text(x + width / 2, -y + 0.80, hold_times_text,
#                     ha='center', va='center', fontsize=6, color='darkblue')
#
#     # Add title and information
#     ax.set_title('Keystroke Dynamics Visualization',
#                  fontsize=14, fontweight='bold', pad=20)
#
#     ax.set_xlim(-0.5, 14.5)
#     ax.set_ylim(-5, 1)
#     ax.set_aspect('equal')
#     ax.axis('off')
#
#     plt.tight_layout()
#     return fig


import plotly.graph_objects as go
import torch
from datetime import datetime

import plotly.graph_objects as go
import torch


def visualize_keystrokes(data, mask, vline_index=None, user=None):
    """
    Visualizes keystroke timing data where keycodes are ASCII values (integers).

    Parameters:
        data: tensor of shape (N, 3) containing:
              data[:, 0]: hold_time (float, in seconds)
              data[:, 1]: flight_time (float, in seconds)
              data[:, 2]: keycode (int, ASCII code, e.g. 104 for 'h')
        mask: tensor of shape (N,) with 1 for valid keystrokes, 0 for padding
    """
    CONTROL_KEY_NAMES = {
        16: "Shift",
        8: "Backspace",
        9: "Tab",
        13: "Enter",
        27: "Escape",
        32: "Space",
        127: "Delete",
    }

    # Apply mask to filter out padding
    valid_indices = (mask != 0)  # True for real keystrokes
    valid_data = data[valid_indices]

    # Extract columns from valid data only
    hold_times = valid_data[:, 0].cpu().numpy()  # (M,)
    flight_times = valid_data[:, 1].cpu().numpy()  # (M,)
    keycodes = valid_data[:, 2].long().cpu().numpy()  # (M,) as integers

    def format_key_label(code):
        code_int = int(code)

        # First, check for named control keys
        if code_int in CONTROL_KEY_NAMES:
            return CONTROL_KEY_NAMES[code_int]

        # Then, printable ASCII
        if 32 <= code_int <= 126:
            return chr(code_int)

        # Fallback: show numeric code
        return f"{code_int}"

    keys = [format_key_label(kc) for kc in keycodes]

    # Calculate statistics for the legend
    num_valid_keystrokes = int(valid_indices.sum().item())
    total_keystrokes = mask.size(0)
    num_padding = total_keystrokes - num_valid_keystrokes

    avg_hold_time = hold_times.mean() if len(hold_times) > 0 else 0
    avg_flight_time = flight_times.mean() if len(flight_times) > 0 else 0
    total_time = (hold_times.sum() + flight_times.sum()) if len(hold_times) > 0 else 0

    num_keys = len(keys)
    base_width = 1000
    min_width = 800
    max_width = 2000


    # Create Plotly figure
    fig = go.Figure()
    x_pos = list(range(len(keys)))

    # Add Hold Time trace
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=hold_times,
        mode='lines+markers+text',
        name=f'Hold Time (avg: {avg_hold_time:.3f}s)',
        marker=dict(color='#2E86AB', size=10),
        line=dict(width=3),
        text=[f'{ht:.3f}' for ht in hold_times],
        textposition='top center',
        textfont=dict(size=15, color='#2E86AB'),
        hovertemplate='Hold: %{y:.3f}s<br>Key: %{text}<extra>Hold Time</extra>',
    ))

    # Add Flight Time trace
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=flight_times,
        mode='lines+markers+text',
        name=f'Flight Time (avg: {avg_flight_time:.3f}s)',
        marker=dict(color='#A23B72', size=10, symbol='square'),
        line=dict(width=3),
        text=[f'{ft:.3f}' for ft in flight_times],
        textposition='bottom center',
        textfont=dict(size=15, color='#A23B72'),
        hovertemplate='Flight: %{y:.3f}s<br>Key: %{text}<extra>Flight Time</extra>',
    ))

    fig.update_xaxes(
        tickmode='array',
        tickvals=x_pos,
        ticktext=keys,
        title=dict(text='Key Pressed', font=dict(size=14)),
        tickfont=dict(size=14)
    )
    # Dynamic width: more keys = wider plot
    dynamic_width = min(max(base_width + num_keys * 20, min_width), max_width)

    # Update layout with enhanced legend and user info
    fig.update_layout(
        title=dict(
            text=f'Keystroke Timing Analysis - User: {user}<br><sub>Total: {num_valid_keystrokes} keystrokes, Padding: {num_padding}</sub>',
            font=dict(size=18),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Key Pressed (Sequence)', font=dict(size=14)),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(text='Time (seconds)', font=dict(size=14)),
            gridcolor='lightgray',
            gridwidth=1
        ),
        legend=dict(
            font=dict(size=12),
            title=dict(
                text=(f'<b>User:</b> {user}  <br>'
                      f'<b>Statistics:</b> <br>'
                      f'• Valid Keystrokes: {num_valid_keystrokes}/{total_keystrokes}<br>'
                      f'• Total Time: {total_time:.3f}s'),
                      # f'• Avg Hold: {avg_hold_time:.3f}s<br>'
                      # f'• Avg Flight: {avg_flight_time:.3f}s'),
                font=dict(size=14, color='#333')
            ),
            bgcolor='rgba(240, 240, 240, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='x unified',
        template='plotly_white',
        width=dynamic_width,  # Dynamic width
        height=600,  # Fixed height or dynamic
        autosize=False,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    if vline_index is not None and 0 <= vline_index < len(x_pos):
        fig.add_shape(
            type="line",
            x0=vline_index,
            x1=vline_index,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",  # spans full height of plot
            line=dict(
                color="red",
                width=3,
                dash="dash"
            )
        )

        # Optional label for the line
        fig.add_annotation(
            x=vline_index,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"Start of Attack {vline_index}",
            showarrow=False,
            font=dict(color="red", size=14)
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = f"keystrokes_{user}_{timestamp}.html"
    # Add grid
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    fig.write_html("XAI_output/"+save_path, include_plotlyjs="cdn")
    fig.show()


def compare_two_users(data1, mask1, user1, data2, mask2, user2, similarity_score):
    """
    Compares keystroke timing data for two users on the same plot.
    Uses visualize_keystrokes internally to generate individual plots and merges them.

    Parameters:
        data1: tensor of shape (N, 3) for user 1
        mask1: tensor of shape (N,) for user 1
        user1: string or identifier for user 1
        data2: tensor of shape (N, 3) for user 2
        mask2: tensor of shape (N,) for user 2
        user2: string or identifier for user 2
        similarity_score: float, similarity score between the two users
    """
    # Generate individual figures for both users
    fig1 = visualize_keystrokes(data1, mask1, user1)
    fig2 = visualize_keystrokes(data2, mask2, user2)

    # Create a new combined figure
    fig = go.Figure()

    # Extract statistics from the original figures
    # User 1 stats
    valid_indices1 = (mask1 != 0)
    num_valid1 = int(valid_indices1.sum().item())
    valid_data1 = data1[valid_indices1]
    hold_times1 = valid_data1[:, 0].cpu().numpy()
    flight_times1 = valid_data1[:, 1].cpu().numpy()
    avg_hold1 = hold_times1.mean() if len(hold_times1) > 0 else 0
    avg_flight1 = flight_times1.mean() if len(flight_times1) > 0 else 0

    # User 2 stats
    valid_indices2 = (mask2 != 0)
    num_valid2 = int(valid_indices2.sum().item())
    valid_data2 = data2[valid_indices2]
    hold_times2 = valid_data2[:, 0].cpu().numpy()
    flight_times2 = valid_data2[:, 1].cpu().numpy()
    avg_hold2 = hold_times2.mean() if len(hold_times2) > 0 else 0
    avg_flight2 = flight_times2.mean() if len(flight_times2) > 0 else 0

    # Add traces from user 1 with modified styling and names
    for trace in fig1.data:
        new_trace = go.Scatter(trace)
        if 'Hold' in trace.name:
            new_trace.name = f'{user1} - Hold Time'
            new_trace.marker.color = '#2E86AB'
            new_trace.marker.size = 8
            new_trace.line.width = 2
            new_trace.mode = 'lines+markers'
            new_trace.text = None
            new_trace.textposition = None
        else:
            new_trace.name = f'{user1} - Flight Time'
            new_trace.marker.color = '#A23B72'
            new_trace.marker.size = 8
            new_trace.line.width = 2
            new_trace.line.dash = 'dash'
            new_trace.mode = 'lines+markers'
            new_trace.text = None
            new_trace.textposition = None
        new_trace.hovertemplate = f'{user1}<br>' + trace.hovertemplate.split('<br>')[1]
        fig.add_trace(new_trace)

    # Add traces from user 2 with modified styling and names
    for trace in fig2.data:
        new_trace = go.Scatter(trace)
        if 'Hold' in trace.name:
            new_trace.name = f'{user2} - Hold Time'
            new_trace.marker.color = '#F18F01'
            new_trace.marker.size = 8
            new_trace.line.width = 2
            new_trace.mode = 'lines+markers'
            new_trace.text = None
            new_trace.textposition = None
        else:
            new_trace.name = f'{user2} - Flight Time'
            new_trace.marker.color = '#06A77D'
            new_trace.marker.size = 8
            new_trace.line.width = 2
            new_trace.line.dash = 'dash'
            new_trace.mode = 'lines+markers'
            new_trace.text = None
            new_trace.textposition = None
        new_trace.hovertemplate = f'{user2}<br>' + trace.hovertemplate.split('<br>')[1]
        fig.add_trace(new_trace)

    # Update layout for comparison
    max_keystrokes = max(num_valid1, num_valid2)
    fig.update_layout(
        title=dict(
            text=f'Keystroke Timing Comparison: <b>{user1}</b> vs <b>{user2}</b><br><sub>Similarity Score: {similarity_score:.4f}</sub>',
            font=dict(size=18),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Keystroke Sequence', font=dict(size=14)),
            tickfont=dict(size=12),
            range=[-0.5, max_keystrokes - 0.5]
        ),
        yaxis=dict(
            title=dict(text='Time (seconds)', font=dict(size=14)),
            gridcolor='lightgray',
            gridwidth=1
        ),
        legend=dict(
            font=dict(size=11),
            title=dict(
                text=(f'<b>Similarity Score: {similarity_score:.4f}</b><br><br>'
                      f'<b>{user1}</b>:<br>'
                      f'• Keystrokes: {num_valid1}<br>'
                      f'• Avg Hold: {avg_hold1:.3f}s<br>'
                      f'• Avg Flight: {avg_flight1:.3f}s<br><br>'
                      f'<b>{user2}</b>:<br>'
                      f'• Keystrokes: {num_valid2}<br>'
                      f'• Avg Hold: {avg_hold2:.3f}s<br>'
                      f'• Avg Flight: {avg_flight2:.3f}s'),
                font=dict(size=10, color='#333')
            ),
            bgcolor='rgba(240, 240, 240, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest',
        template='plotly_white',
        width=1200,
        height=600,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    fig.show()


def visualize_activations(net, datamodule, color="C0"):
    """
    Visualize activations throughout the network by registering forward hooks.
    Adapted from the PyTorch Lightning UvA Deep Learning course.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    activations = {}
    # Hook function to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for layers with parameters or specific activation types
    hooks = []
    for name, module in net.named_modules():
        if hasattr(module, 'weight') or isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh,
                                                            nn.Sigmoid, nn.GELU, nn.BatchNorm1d,
                                                            nn.BatchNorm2d, nn.LayerNorm, LearnableFourierFeatures)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Get a batch of data and run forward pass
    small_loader = datamodule.train_dataloader()
    (x1, x2), labels, (u1, u2) = next(iter(small_loader))

    with torch.no_grad():
        _ = net(x1.float().to(device))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process activations for plotting
    processed_activations = {name: activation.view(-1).cpu().numpy()
                            for name, activation in activations.items()}

    # Create subplot grid
    columns = 3
    rows = math.ceil(len(processed_activations) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3))
    axes = np.atleast_2d(axes)

    # Plot each layer's activations
    for idx, (name, activation_np) in enumerate(processed_activations.items()):
        row, col = idx // columns, idx % columns
        ax = axes[row, col]

        sns.histplot(data=activation_np, bins=50, ax=ax, color=color, kde=True, stat="density")
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)

        display_name = name.replace('model.', '')
        module_type = type(dict(net.named_modules())[name]).__name__
        ax.set_title(f"{display_name}\n({module_type})", fontsize=10)

        stats_text = f"μ={activation_np.mean():.2f}, σ={activation_np.std():.2f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8)

    # Turn off unused subplots
    for idx in range(len(processed_activations), rows * columns):
        row, col = idx // columns, idx % columns
        axes[row, col].axis('off')

    fig.suptitle("Activation distributions", fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
    plt.savefig("activations_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return processed_activations

# Example usage
if __name__ == "__main__":
    # Example tensor corresponding to typing "hello"
    # Columns: hold_time, flight_time, keycode (ASCII)
    data = torch.tensor([
        [0.15, 0.30, ord('H')],  # Real keystroke
        [0.12, 0.25, ord('e')],  # Real keystroke
        [0.10, 0.20, ord('l')],  # Real keystroke
        [0.11, 0.22, ord('l')],  # Real keystroke
        [0.20, 0.35, ord('o')],  # Real keystroke
        [0.00, 0.00, 0],  # Padding
        [0.00, 0.00, 0],  # Padding
        [0.00, 0.00, 0]  # Padding
    ])

    # mask shape: (8,) with 1 for valid, 0 for padding
    mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])
    #
    # # Visualize
    # visualize_keystrokes(data, mask, 'azeaze')

    compare_two_users(data+0.2, mask, 'Diana', data, mask, 'Igor', 2.0)


