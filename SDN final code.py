import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

plt.style.use('seaborn-v0_8')

print()
print("Project Title --- SDN BASED ADAPTIVE PROTOCOL ROUTING USING OPENFLOW")
print()

# =========================================================
# TOPOLOGY (Mininet-like)
# =========================================================

def Mininet():
    G = nx.Graph()

    G.add_node('Controller1', layer='Controller', ip='192.168.0.1')
    G.add_node('Controller2', layer='Controller', ip='192.168.0.2')

    G.add_node('Switch1', layer='Switch', ip='192.168.1.1')
    G.add_node('Switch2', layer='Switch', ip='192.168.1.2')
    G.add_node('Switch3', layer='Switch', ip='192.168.2.1')
    G.add_node('Switch4', layer='Switch', ip='192.168.2.2')

    for i in range(1, 5):
        G.add_node(f'Host{i}', layer='Host', ip=f'192.168.1.{10 + i}')
    for i in range(5, 9):
        G.add_node(f'Host{i}', layer='Host', ip=f'192.168.2.{10 + i}')

    G.add_edge('Controller1', 'Switch1')
    G.add_edge('Controller1', 'Switch2')
    G.add_edge('Controller2', 'Switch3')
    G.add_edge('Controller2', 'Switch4')

    G.add_edge('Controller1', 'Controller2')

    G.add_edge('Switch1', 'Host1')
    G.add_edge('Switch1', 'Host2')
    G.add_edge('Switch2', 'Host3')
    G.add_edge('Switch2', 'Host4')
    G.add_edge('Switch3', 'Host5')
    G.add_edge('Switch3', 'Host6')
    G.add_edge('Switch4', 'Host7')
    G.add_edge('Switch4', 'Host8')

    G.add_edge('Switch1', 'Switch2')
    G.add_edge('Switch2', 'Switch3')
    G.add_edge('Switch3', 'Switch4')
    G.add_edge('Switch1', 'Switch3')
    G.add_edge('Switch2', 'Switch4')

    return G

def assign_link_parameters(G, seed=None):
    # if seed is None, parameters will change each run
    rng = np.random.RandomState(seed)
    for u, v in G.edges():
        G[u][v]['cost'] = rng.randint(1, 11)
        G[u][v]['capacity'] = rng.uniform(100, 1000)
        G[u][v]['load'] = rng.uniform(0.1, 0.95)
        G[u][v]['delay'] = 3 + G[u][v]['load'] * 25 + rng.normal(0, 2)
        G[u][v]['packet_loss'] = 0.2 + G[u][v]['load'] * 2.5 + rng.normal(0, 0.3)
        G[u][v]['jitter'] = abs(rng.normal(1.5, 0.5))

def total_path_cost(G, path):
    return sum(G[path[i]][path[i+1]].get('cost', 1) for i in range(len(path) - 1))

# =========================================================
# STATIC TOPOLOGY (with highlighted path)
# =========================================================

def draw_network(G, path_edges):
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(8, 6))

    controller_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'Controller']
    switch_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'Switch']
    host_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'Host']

    nx.draw_networkx_nodes(G, pos, nodelist=controller_nodes,
                           node_color='red', node_size=600, edgecolors='black', label='Controllers')
    nx.draw_networkx_nodes(G, pos, nodelist=switch_nodes,
                           node_color='blue', node_size=500, edgecolors='black', label='Switches')
    nx.draw_networkx_nodes(G, pos, nodelist=host_nodes,
                           node_color='green', node_size=400, edgecolors='black', label='Hosts')

    all_edges = set(G.edges())
    path_edges_set = set(path_edges)
    nx.draw_networkx_edges(G, pos, edgelist=list(all_edges - path_edges_set),
                           width=2.0, edge_color='gray', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           width=3.0, edge_color='orange', style='dashed')

    edge_labels = {(u, v): G[u][v].get('cost', '') for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.title("Network Topology - Shortest Path Highlighted")
    plt.legend(loc='best')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# =========================================================
# ROUTING METRICS (values depend on path)
# =========================================================

def routing_traditional_metrics(G, path_edges):
    delays = []
    capacities = []
    loads = []
    losses = []
    jitters = []

    for (u, v) in path_edges:
        e = G[u][v]
        capacities.append(e['capacity'])
        loads.append(e['load'])
        delays.append(e['delay'])
        losses.append(e['packet_loss'])
        jitters.append(e['jitter'])

    capacities = np.array(capacities)
    loads = np.array(loads)
    delays = np.array(delays)
    losses = np.array(losses)
    jitters = np.array(jitters)

    avg_delay = delays.mean()
    avg_throughput = (capacities * (1 - loads)).mean()
    avg_loss = losses.mean()
    avg_jitter = jitters.mean()
    avg_util = loads.mean()

    qos = (avg_throughput / capacities.max()) * 0.4 \
        + (1 / (1 + avg_delay)) * 0.3 \
        + (1 / (1 + avg_loss)) * 0.2 \
        + (1 / (1 + avg_jitter)) * 0.1

    return {
        'Method': 'Traditional',
        'Delay_ms': avg_delay,
        'Throughput_Mbps': avg_throughput,
        'PacketLoss_percent': avg_loss,
        'Jitter_ms': avg_jitter,
        'Utilization': avg_util,
        'QoS_score': qos
    }

def routing_sdn_metrics(G, path_edges):
    delays = []
    capacities = []
    loads = []
    losses = []
    jitters = []
    weights = []

    for (u, v) in path_edges:
        e = G[u][v]
        capacities.append(e['capacity'])
        loads.append(e['load'])
        delays.append(e['delay'])
        losses.append(e['packet_loss'])
        jitters.append(e['jitter'])
        w = 1.0 / (1.0 + e['delay'] + e['packet_loss'])
        weights.append(w)

    capacities = np.array(capacities)
    loads = np.array(loads)
    delays = np.array(delays)
    losses = np.array(losses)
    jitters = np.array(jitters)
    weights = np.array(weights)
    weights = weights / weights.sum()

    avg_delay = np.sum(delays * weights)
    avg_throughput = np.sum(capacities * (1 - loads) * weights)
    avg_loss = np.sum(losses * weights)
    avg_jitter = np.sum(jitters * weights)
    avg_util = np.sum(loads * weights)

    qos = (avg_throughput / capacities.max()) * 0.4 \
        + (1 / (1 + avg_delay)) * 0.3 \
        + (1 / (1 + avg_loss)) * 0.2 \
        + (1 / (1 + avg_jitter)) * 0.1

    return {
        'Method': 'SDN Adaptive',
        'Delay_ms': avg_delay,
        'Throughput_Mbps': avg_throughput,
        'PacketLoss_percent': avg_loss,
        'Jitter_ms': avg_jitter,
        'Utilization': avg_util,
        'QoS_score': qos
    }

# -------- different graph types (bar, pie, waterfall, line, stacked, scatter) --------

def plot_delay_bar_vertical(routing_df):
    plt.figure(figsize=(6, 4))
    methods = routing_df['Method']
    values = routing_df['Delay_ms'].values
    colors = ['#264653', '#e76f51']
    plt.bar(methods, values, color=colors)
    for i, v in enumerate(values):
        plt.text(i, v * 1.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    plt.ylabel('Delay (ms)')
    plt.title('Average Delay – Traditional vs SDN')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_throughput_pie(routing_df):
    plt.figure(figsize=(5, 5))
    values = routing_df['Throughput_Mbps'].values
    labels = routing_df['Method']
    colors = ['#2a9d8f', '#f4a261']
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Share of Throughput – Traditional vs SDN')
    plt.tight_layout()
    plt.show()

def plot_packet_loss_waterfall(routing_df):
    plt.figure(figsize=(6, 4))
    base = routing_df.loc[routing_df['Method']=='Traditional','PacketLoss_percent'].values[0]
    sdn_val = routing_df.loc[routing_df['Method']=='SDN Adaptive','PacketLoss_percent'].values[0]
    diff = sdn_val - base

    labels = ['Traditional', 'Change', 'SDN Adaptive']
    values = [base, diff, sdn_val]
    cumulative = [0, base, base + diff]

    for i in range(len(values)):
        if i == 0 or i == 2:
            color = '#457b9d'
        else:
            color = '#e63946' if diff > 0 else '#2a9d8f'
        plt.bar(labels[i], values[i], bottom=cumulative[i], color=color)
        plt.text(i, cumulative[i] + values[i] * 1.01,
                 f'{cumulative[i] + values[i]:.2f}', ha='center', fontsize=8)

    plt.ylabel('Packet Loss (%)')
    plt.title('Packet Loss – Waterfall (Traditional → SDN)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_jitter_line_markers(routing_df):
    plt.figure(figsize=(6, 4))
    x = [0, 1]
    values = routing_df['Jitter_ms'].values
    colors = ['#ff006e', '#8338ec']
    plt.plot(x, values, color='#adb5bd', linewidth=1.5)
    plt.scatter(x, values, c=colors, s=80)
    for i, v in enumerate(values):
        plt.text(x[i], v * 1.05, f'{v:.2f}', ha='center', fontsize=9)
    plt.xticks(x, routing_df['Method'])
    plt.ylabel('Jitter (ms)')
    plt.title('Jitter – Traditional vs SDN')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_utilization_stacked(routing_df):
    plt.figure(figsize=(6, 4))
    trad = routing_df.loc[routing_df['Method']=='Traditional','Utilization'].values[0]
    sdn = routing_df.loc[routing_df['Method']=='SDN Adaptive','Utilization'].values[0]
    plt.bar(['Average Utilization'], [trad], color='#118ab2', label='Traditional')
    plt.bar(['Average Utilization'], [sdn], bottom=[trad], color='#ffd166', label='SDN Adaptive')
    plt.ylabel('Utilization (stacked)')
    plt.title('Link Utilization – Stacked View')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_qos_scatter(routing_df):
    plt.figure(figsize=(6, 4))
    methods = routing_df['Method']
    values = routing_df['QoS_score'].values
    colors = ['#06d6a0', '#ef476f']
    for m, v, c in zip(methods, values, colors):
        plt.scatter(m, v, color=c, s=100)
        plt.text(m, v * 1.03, f'{v:.3f}', ha='center', fontsize=9)
    plt.ylabel('QoS Score')
    plt.title('QoS – Traditional vs SDN')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================================================
# ATTACK MODULE (if you need it, keep your existing code)
# =========================================================

# ... (attack functions here, unchanged) ...

# =========================================================
# MAIN
# =========================================================

def main(G):
    # if you want repeatable runs, pass a fixed seed (e.g., 0)
    assign_link_parameters(G, seed=None)

    source = input("Enter the source node (e.g., Host1): ")
    destination = input("Enter the destination node (e.g., Host7): ")

    if source not in G.nodes or destination not in G.nodes:
        print("Invalid source or destination.")
        return

    all_paths = list(nx.all_simple_paths(G, source=source, target=destination))
    if not all_paths:
        print("No path found between the specified nodes.")
        return

    print("\nAll possible paths (with total cost):")
    path_costs = []
    for idx, p in enumerate(all_paths, start=1):
        c = total_path_cost(G, p)
        path_costs.append((p, c))
        print(f"Path {idx}: {' -> '.join(p)} | Total cost: {c} ms")

    best_path, best_cost = min(path_costs, key=lambda x: x[1])
    print("\nSelected Shortest (Min-Cost) Path:")
    for i in range(len(best_path) - 1):
        print(f"{best_path[i]} -> {best_path[i+1]}")
    print(f"Total path cost: {best_cost} ms")
    path_edges = list(zip(best_path, best_path[1:]))

    # topology with highlighted shortest path
    draw_network(G, path_edges)

    # routing metrics
    trad_route = routing_traditional_metrics(G, path_edges)
    sdn_route = routing_sdn_metrics(G, path_edges)
    routing_df = pd.DataFrame([trad_route, sdn_route])
    print("\nRouting metrics:\n", routing_df)

    # six different graph styles
    plot_delay_bar_vertical(routing_df)
    plot_throughput_pie(routing_df)
    plot_packet_loss_waterfall(routing_df)
    plot_jitter_line_markers(routing_df)
    plot_utilization_stacked(routing_df)
    plot_qos_scatter(routing_df)

if __name__ == "__main__":
    G = Mininet()
    main(G)