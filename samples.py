import os
from seisclass import check_seed


test_seed = os.path.join(os.path.dirname(__file__), 'samples', 'test.seed')
test_phase = os.path.join(os.path.dirname(__file__), 'samples', 'test.phase')

print("=== seisclass Library Usage Example ===")
print(f"SEED file path: {test_seed}")
print(f"Phase file path: {test_phase}")
print()

result = check_seed(test_seed, test_phase)

print("=== Classification Result ===")
print(f"Classification result: {result}")
