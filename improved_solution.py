import csv
import os
import heapq
from collections import defaultdict, deque
from tqdm import tqdm
import time
import math
import random

def read_all_data(input_dir):
    graph = {}
    transfer_price = {}
    transfer_max = {}
    requests = []
    
    print(f"  Чтение данных из {input_dir}...")
    
    distance_file = os.path.join(input_dir, 'distance_matrix.csv')
    if os.path.exists(distance_file):
        with open(distance_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                a, b = int(row[0]), int(row[1])
                if a not in graph:
                    graph[a] = {}
                graph[a][b] = float(row[5])
    
    reqs_file = os.path.join(input_dir, 'reqs.csv')
    if os.path.exists(reqs_file):
        with open(reqs_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                requests.append({
                    'src': int(row[0]),
                    'dst': int(row[1]),
                    'volume': float(row[2])
                })
    
    offices_file = os.path.join(input_dir, 'offices.csv')
    if os.path.exists(offices_file):
        with open(offices_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                node = int(row[2])
                transfer_price[node] = float(row[3])
                transfer_max[node] = float(row[4])
    
    return graph, requests, transfer_price, transfer_max

def yen_k_shortest_paths(graph, start, end, k=5):
    try:
        def dijkstra():
            dist = {start: 0}
            prev = {}
            pq = [(0, start)]
            
            while pq:
                current_dist, u = heapq.heappop(pq)
                if u == end:
                    break
                if u not in graph:
                    continue
                for v, cost in graph[u].items():
                    new_dist = current_dist + cost
                    if v not in dist or new_dist < dist[v]:
                        dist[v] = new_dist
                        prev[v] = u
                        heapq.heappush(pq, (new_dist, v))
            
            if end not in dist:
                return None
            path = []
            current = end
            while current != start:
                path.append(current)
                current = prev[current]
            path.append(start)
            path.reverse()
            return path

        A = [dijkstra()]
        if not A[0]:
            return []
        
        B = []
        
        for ki in range(1, k):
            for i in range(len(A[-1]) - 1):
                spurNode = A[-1][i]
                rootPath = A[-1][:i+1]
                
                edges_removed = []
                for path in A:
                    if len(path) > i and rootPath == path[:i+1]:
                        u = path[i]
                        v = path[i+1] if i+1 < len(path) else None
                        if v and u in graph and v in graph[u]:
                            cost = graph[u][v]
                            del graph[u][v]
                            edges_removed.append((u, v, cost))
                
                spurPath = dijkstra_from_node(graph, spurNode, end)
                
                for u, v, cost in edges_removed:
                    if u not in graph:
                        graph[u] = {}
                    graph[u][v] = cost
                
                if spurPath:
                    totalPath = rootPath[:-1] + spurPath
                    if totalPath not in B:
                        B.append(totalPath)
            
            if not B:
                break
                
            B.sort(key=lambda p: sum(graph[p[i]][p[i+1]] for i in range(len(p)-1)))
            A.append(B.pop(0))
        
        return A[:k]
    except:
        return find_simple_paths(graph, start, end, k)

def dijkstra_from_node(graph, start, end):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if u == end:
            break
        if u not in graph:
            continue
        for v, cost in graph[u].items():
            new_dist = current_dist + cost
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    if end not in dist:
        return None
    path = []
    current = end
    while current != start:
        path.append(current)
        current = prev.get(current)
        if current is None:
            return None
    path.append(start)
    path.reverse()
    return path

def find_simple_paths(graph, start, end, k=5, max_depth=10):
    paths = []
    
    if start in graph and end in graph[start]:
        paths.append([start, end])
    
    queue = deque([([start], 0)])
    visited_paths = set()
    
    while queue and len(paths) < k * 3:
        path, cost = queue.popleft()
        current = path[-1]
        
        if current == end and len(path) > 1:
            path_tuple = tuple(path)
            if path_tuple not in visited_paths:
                paths.append(path)
                visited_paths.add(path_tuple)
            continue
        
        if len(path) >= max_depth:
            continue
            
        if current in graph:
            for neighbor, edge_cost in graph[current].items():
                if neighbor not in path:
                    new_cost = cost + edge_cost
                    new_path = path + [neighbor]
                    queue.append((new_path, new_cost))
    
    def path_cost(p):
        try:
            return sum(graph[p[i]][p[i+1]] for i in range(len(p)-1))
        except:
            return float('inf')
    
    paths.sort(key=lambda p: (path_cost(p), len(p)))
    return paths[:k]

class TruckAllocator:
    def __init__(self, graph, C=90):
        self.graph = graph
        self.C = C
        self.edge_flows = defaultdict(float)
        self.edge_trucks = defaultdict(int)
    
    def add_flow(self, edge, volume):
        a, b = edge
        old_volume = self.edge_flows[edge]
        new_volume = old_volume + volume
        
        old_trucks = math.ceil(old_volume / self.C)
        new_trucks = math.ceil(new_volume / self.C)
        
        self.edge_flows[edge] = new_volume
        self.edge_trucks[edge] = new_trucks
    
    def get_edge_cost(self, edge, additional_volume=0):
        a, b = edge
        current_volume = self.edge_flows.get(edge, 0)
        total_volume = current_volume + additional_volume
        
        current_trucks = math.ceil(current_volume / self.C)
        new_trucks = math.ceil(total_volume / self.C)
        
        additional_trucks = new_trucks - current_trucks
        return additional_trucks * self.graph[a][b]

def adaptive_transfer_limit(graph_size, volume, transfer_max_node, current_usage):
    base_limit = transfer_max_node
    
    if graph_size < 30:
        safety_margin = 0.85
        penalty_multiplier = 8
    elif graph_size < 100:
        safety_margin = 0.90
        penalty_multiplier = 5
    else:
        safety_margin = 0.95
        penalty_multiplier = 3
    
    hard_limit = base_limit * safety_margin
    soft_limit = base_limit * 0.98
    
    return hard_limit, soft_limit, penalty_multiplier

def calculate_path_cost_comprehensive(path, volume, graph, transfer_price, truck_allocator, graph_size, node_usage, transfer_max):
    total_cost = 0
    
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        edge_cost = truck_allocator.get_edge_cost(edge, volume)
        total_cost += edge_cost
    
    transfer_cost = 0
    overload_penalty = 0
    
    for i in range(1, len(path) - 1):
        node = path[i]
        if node != path[0] and node != path[-1]:
            transfer_cost += transfer_price.get(node, 0) * volume
            
            hard_limit, soft_limit, penalty_multiplier = adaptive_transfer_limit(
                graph_size, volume, transfer_max.get(node, float('inf')), node_usage.get(node, 0)
            )
            
            predicted_usage = node_usage.get(node, 0) + volume
            if predicted_usage > soft_limit:
                overload = predicted_usage - soft_limit
                overload_penalty += overload * transfer_price.get(node, 0) * penalty_multiplier
    
    total_cost += transfer_cost + overload_penalty
    
    if graph_size > 50 and len(path) > 6:
        total_cost *= (1 + 0.05 * (len(path) - 6))
    
    return total_cost

def cluster_requests_by_volume_distance(requests, graph):
    if not requests:
        return requests
    
    complexities = []
    for req in requests:
        src, dst, volume = req['src'], req['dst'], req['volume']
        direct_cost = graph.get(src, {}).get(dst, 1000)
        complexity = volume * direct_cost if direct_cost > 0 else volume * 1000
        complexities.append((complexity, req))
    
    complexities.sort(key=lambda x: x[0], reverse=True)
    
    n = len(complexities)
    complex_reqs = [req for _, req in complexities[:n//3]]
    medium_reqs = [req for _, req in complexities[n//3:2*n//3]]
    simple_reqs = [req for _, req in complexities[2*n//3:]]
    
    result = []
    max_len = max(len(complex_reqs), len(medium_reqs), len(simple_reqs))
    
    for i in range(max_len):
        if i < len(complex_reqs):
            result.append(complex_reqs[i])
        if i < len(medium_reqs):
            result.append(medium_reqs[i])
        if i < len(simple_reqs):
            result.append(simple_reqs[i])
    
    return result

def find_optimal_path_for_request(req, graph, transfer_price, transfer_max, truck_allocator, node_usage, graph_size, k_paths=10):
    src, dst, volume = req['src'], req['dst'], req['volume']
    
    if graph_size < 30:
        k_paths = min(8, k_paths)
    elif graph_size > 100:
        k_paths = min(15, k_paths)
    
    candidate_paths = []
    
    if src in graph and dst in graph[src]:
        candidate_paths.append([src, dst])
    
    k_shortest = yen_k_shortest_paths(graph, src, dst, k_paths)
    for path in k_shortest:
        if path and path not in candidate_paths:
            candidate_paths.append(path)
    
    if graph_size > 50 and len(candidate_paths) < 5:
        additional_paths = find_simple_paths(graph, src, dst, 5)
        for path in additional_paths:
            if path and path not in candidate_paths:
                candidate_paths.append(path)
    
    path_costs = []
    
    for path in candidate_paths:
        if not path:
            continue
            
        valid = True
        total_overload_penalty = 0
        
        for i in range(1, len(path) - 1):
            node = path[i]
            if node != src and node != dst:
                hard_limit, soft_limit, penalty_multiplier = adaptive_transfer_limit(
                    graph_size, volume, transfer_max.get(node, float('inf')), node_usage.get(node, 0)
                )
                
                predicted_usage = node_usage.get(node, 0) + volume
                if predicted_usage > hard_limit:
                    valid = False
                    break
                elif predicted_usage > soft_limit:
                    overload = predicted_usage - soft_limit
                    total_overload_penalty += overload * transfer_price.get(node, 0) * penalty_multiplier
        
        if valid:
            cost = calculate_path_cost_comprehensive(
                path, volume, graph, transfer_price, truck_allocator, graph_size, node_usage, transfer_max
            )
            cost += total_overload_penalty
            path_costs.append((cost, path))
    
    if not path_costs:
        for path in candidate_paths:
            if not path:
                continue
                
            cost = calculate_path_cost_comprehensive(
                path, volume, graph, transfer_price, truck_allocator, graph_size, node_usage, transfer_max
            )
            
            overload_penalty = 0
            for i in range(1, len(path) - 1):
                node = path[i]
                if node != src and node != dst:
                    hard_limit, _, _ = adaptive_transfer_limit(
                        graph_size, volume, transfer_max.get(node, float('inf')), node_usage.get(node, 0)
                    )
                    overload = max(0, node_usage.get(node, 0) + volume - hard_limit)
                    overload_penalty += overload * transfer_price.get(node, 0) * 20
            
            path_costs.append((cost + overload_penalty, path))
    
    if not path_costs:
        for path in candidate_paths:
            if path:
                cost = calculate_path_cost_comprehensive(
                    path, volume, graph, transfer_price, truck_allocator, graph_size, node_usage, transfer_max
                )
                path_costs.append((cost, path))
    
    if not path_costs:
        return None
    
    path_costs.sort(key=lambda x: x[0])
    return path_costs[0][1]

def optimize_with_adaptive_strategy(requests, graph, transfer_price, transfer_max, max_iterations=4):
    best_solutions = None
    best_cost = float('inf')
    
    graph_size = len(graph)
    print(f"      Размер графа: {graph_size} узлов")
    
    for iteration in range(max_iterations):
        print(f"    Итерация {iteration + 1}/{max_iterations}")
        
        truck_allocator = TruckAllocator(graph)
        node_usage = defaultdict(float)
        solutions = []
        
        if iteration == 0:
            sorted_requests = sorted(requests, key=lambda x: x['volume'] * 
                                   graph.get(x['src'], {}).get(x['dst'], 1000), reverse=True)
        elif iteration == 1:
            sorted_requests = cluster_requests_by_volume_distance(requests, graph)
        elif iteration == 2:
            sorted_requests = requests.copy()
            random.shuffle(sorted_requests)
        else:
            sorted_requests = list(reversed(requests))
        
        if graph_size < 30:
            k_paths = 6
        elif graph_size < 100:
            k_paths = 8
        else:
            k_paths = 12
        
        successful_requests = 0
        total_requests = len(sorted_requests)
        
        for req in tqdm(sorted_requests, desc=f"      Итерация {iteration+1}", leave=False):
            path = find_optimal_path_for_request(
                req, graph, transfer_price, transfer_max, truck_allocator, 
                node_usage, graph_size, k_paths
            )
            
            if path:
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    truck_allocator.add_flow(edge, req['volume'])
                
                for i in range(1, len(path) - 1):
                    node = path[i]
                    if node != req['src'] and node != req['dst']:
                        node_usage[node] += req['volume']
                
                solutions.append({
                    'src': req['src'],
                    'dst': req['dst'],
                    'volume': req['volume'],
                    'path': path
                })
                successful_requests += 1
        
        total_cost = 0
        
        for edge, trucks in truck_allocator.edge_trucks.items():
            a, b = edge
            total_cost += trucks * graph[a][b]
        
        transit_usage = defaultdict(float)
        for sol in solutions:
            path = sol['path']
            for i in range(1, len(path) - 1):
                node = path[i]
                if node != sol['src'] and node != sol['dst']:
                    transit_usage[node] += sol['volume']
        
        for node, usage in transit_usage.items():
            total_cost += usage * transfer_price.get(node, 0)
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        print(f"      Стоимость: {total_cost:,.0f}, Успешно: {success_rate:.1%}")
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_solutions = solutions.copy()
            print(f"      Новый лучший результат!")
    
    return best_solutions

def generate_high_quality_solution(input_dir, output_file):
    graph, requests, transfer_price, transfer_max = read_all_data(input_dir)
    
    print(f"  Обработка {len(requests)} заказов...")
    print(f"  Узлы с ограничениями: {len(transfer_max)}")
    
    total_volume = sum(req['volume'] for req in requests)
    if total_volume < 1000:
        max_iterations = 3
    elif total_volume < 5000:
        max_iterations = 4
    else:
        max_iterations = 5
    
    solutions = optimize_with_adaptive_strategy(
        requests, graph, transfer_price, transfer_max, max_iterations
    )
    
    print("  Сохранение финального решения...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst', 'volume', 'path_nodes'])
        
        for sol in solutions:
            writer.writerow([
                sol['src'],
                sol['dst'],
                round(sol['volume'], 6),
                str(sol['path']).replace(' ', '')
            ])
    
    truck_allocator = TruckAllocator(graph)
    transit_usage = defaultdict(float)
    
    for sol in solutions:
        for i in range(len(sol['path']) - 1):
            edge = (sol['path'][i], sol['path'][i+1])
            truck_allocator.add_flow(edge, sol['volume'])
        for i in range(1, len(sol['path']) - 1):
            node = sol['path'][i]
            if node != sol['src'] and node != sol['dst']:
                transit_usage[node] += sol['volume']
    
    total_truck_cost = sum(trucks * graph[a][b] for (a, b), trucks in truck_allocator.edge_trucks.items())
    total_transfer_cost = sum(usage * transfer_price.get(node, 0) for node, usage in transit_usage.items())
    total_cost = total_truck_cost + total_transfer_cost
    
    constraint_violations = 0
    for node, capacity in transfer_max.items():
        if transit_usage.get(node, 0) > capacity:
            constraint_violations += 1
    
    print(f"  Решение сохранено: {output_file}")
    print(f"  Обработано заказов: {len(solutions)}/{len(requests)} ({len(solutions)/len(requests):.1%})")
    print(f"  Финальная стоимость: {total_cost:,.0f}")
    print(f"  Стоимость грузовиков: {total_truck_cost:,.0f}")
    print(f"  Стоимость перегрузки: {total_transfer_cost:,.0f}")
    print(f"  Эффективность грузовиков: {sum(truck_allocator.edge_flows.values()) / (90 * sum(truck_allocator.edge_trucks.values())):.1%}")
    print(f"  Нарушения ограничений: {constraint_violations}")
    
    return len(solutions)

def process_all_datasets():
    datasets = ['10', '20', '50', '140']
    
    total_start = time.time()
    
    for dataset in tqdm(datasets, desc="Обработка наборов данных"):
        dataset_start = time.time()
        
        input_dir = os.path.join("data", dataset)
        output_csv = f"solutions/solution_{dataset}.csv"
        
        if os.path.exists(input_dir):
            print(f"\n{'='*60}")
            print(f"НАБОР ДАННЫХ: {dataset}")
            print(f"{'='*60}")
            
            generate_high_quality_solution(input_dir, output_csv)
            
            dataset_time = time.time() - dataset_start
            print(f"  Время обработки: {dataset_time:.1f} сек")
        else:
            print(f"Пропускаем {dataset}: папка {input_dir} не найдена")
    
    total_time = time.time() - total_start
    print(f"\nВсе наборы обработаны за {total_time:.1f} сек")

if __name__ == "__main__":
    os.makedirs("solutions", exist_ok=True)
    process_all_datasets()