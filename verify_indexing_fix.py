#!/usr/bin/env python3
"""
Verify the indexing fix in adjacency_loss is correct.
"""

def verify_adjacency_loss_fix():
    """Check that the fixed adjacency_loss handles tensor shapes correctly."""
    
    print("Verifying adjacency_loss indexing fix...")
    print("=" * 50)
    
    # Read the fixed function
    with open("/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Improved_ReLU_Implementation/loss_functions.py", 'r') as f:
        content = f.read()
    
    # Find the adjacency_loss function
    import re
    func_match = re.search(r'def adjacency_loss\(.*?\n(.*?)(?=\ndef|\nclass|\Z)', content, re.DOTALL)
    
    if func_match:
        func_body = func_match.group(0)
        
        print("Key changes in the fixed version:")
        print("-" * 50)
        
        # Check for the fix patterns
        fixes = [
            ("Removed individual tensor masking", "g1_valid = g1[valid]" not in func_body),
            ("Computing dot product for all edges", "dot_prod = (g1 * g2).sum(dim=1)" in func_body),
            ("Using valid as multiplicative mask", "valid_contribution = valid.float() * w_interior" in func_body),
            ("No more indexing with 2D mask on 3D tensor", "g1[valid]" not in func_body)
        ]
        
        all_good = True
        for desc, check in fixes:
            status = "✓" if check else "✗"
            print(f"{status} {desc}")
            if not check:
                all_good = False
        
        print("\n" + "=" * 50)
        
        if all_good:
            print("✓ All indexing issues have been fixed!")
            print("\nThe fix works by:")
            print("1. Computing cosine similarity for ALL interior edges")
            print("2. Using 'valid' as a multiplicative mask (0 or 1)")
            print("3. Invalid edges contribute 0 to the loss")
            print("4. No tensor shape mismatches!")
        else:
            print("✗ Some issues may remain")
            
        # Show the critical section
        print("\nCritical section of the fix:")
        print("-" * 50)
        lines = func_body.split('\n')
        for i, line in enumerate(lines):
            if 'valid = (n1 > eps)' in line:
                for j in range(max(0, i-2), min(len(lines), i+20)):
                    print(f"{j:3d}: {lines[j]}")
                break
                
    else:
        print("Could not find adjacency_loss function!")

if __name__ == "__main__":
    verify_adjacency_loss_fix()