import csv
import os

def read_graph(input_dir):
    """Читаем граф дорог"""
    graph = {}
    with open(os.path.join(input_dir, 'distance_matrix.csv'), 'r') as file:
        reader = csv.reader(file)
        next(reader)  # пропускаем заголовок
        for row in reader:
            a, b = int(row[0]), int(row[1])
            if a not in graph:
                graph[a] = {}
            graph[a][b] = float(row[5])
    return graph

def read_requests(input_dir):
    """Читаем заказы"""
    requests = []
    with open(os.path.join(input_dir, 'reqs.csv'), 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            requests.append({
                'src': int(row[0]),
                'dst': int(row[1]),
                'volume': float(row[2])
            })
    return requests

def find_direct_path(graph, src, dst):
    """Ищем прямой путь от src к dst"""
    if src in graph and dst in graph[src]:
        return [src, dst]
    return None

def generate_baseline_solution(input_dir, output_file):
    """Генерируем бейзлайн решение - отправляем напрямую если возможно"""
    graph = read_graph(input_dir)
    requests = read_requests(input_dir)
    
    solutions = []
    
    for req in requests:
        src, dst, volume = req['src'], req['dst'], req['volume']
        
        # Пытаемся найти прямой путь
        direct_path = find_direct_path(graph, src, dst)
        
        if direct_path:
            # Если прямой путь существует - используем его
            solutions.append({
                'src': src,
                'dst': dst,
                'volume': volume,
                'path': direct_path
            })
        else:
            # Если прямого пути нет - нужно найти любой путь
            # Пока просто используем первый найденный путь (простейшая реализация)
            path = find_any_path(graph, src, dst)
            if path:
                solutions.append({
                    'src': src,
                    'dst': dst,
                    'volume': volume,
                    'path': path
                })
            else:
                print(f"Внимание: не найден путь от {src} до {dst}")
    
    # Записываем решение в файл
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst', 'volume', 'path_nodes'])
        for sol in solutions:
            writer.writerow([
                sol['src'],
                sol['dst'], 
                sol['volume'],
                str(sol['path']).replace(' ', '')
            ])
    
    print(f"Бейзлайн решение сохранено в: {output_file}")
    return len(solutions)

def find_any_path(graph, start, end, path=[]):
    """Находит любой путь от start до end (рекурсивный поиск в глубину)"""
    path = path + [start]
    
    if start == end:
        return path
        
    if start not in graph:
        return None
        
    for node in graph[start]:
        if node not in path:
            newpath = find_any_path(graph, node, end, path)
            if newpath:
                return newpath
    return None

if __name__ == "__main__":
    input_dir = "data"
    output_file = "solutions/baseline_solution.csv"
    os.makedirs("solutions", exist_ok=True)
    generate_baseline_solution(input_dir, output_file)