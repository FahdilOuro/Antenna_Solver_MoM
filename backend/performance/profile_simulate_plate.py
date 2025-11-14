import cProfile
import pstats
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Optional profilers - gracefully handle if not installed
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not installed. Memory profiling disabled.")

try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    print("Warning: line_profiler not installed. Line-by-line profiling disabled.")


class SimulationProfiler:
    """
    Comprehensive profiling suite for simulate_plate_main.py
    """
    
    def __init__(self, target_file="backend/antenna_simulation/scattering_algorithm_example_implementation/simulate_plate_main.py"):
        self.target_file = Path(target_file)
        self.output_dir = Path("backend/performance/profiling_results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.module = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def timer(self, label):
        """Context manager for timing operations"""
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            print(f"Completed in {elapsed:.3f} seconds")
    
    def import_module(self):
        """
        Import the simulation module with fallback strategies
        """
        if self.module is not None:
            return self.module
        
        print("Importing simulation module...")
        
        # Strategy 1: Direct import
        try:
            from antenna_simulation.scattering_algorithm_example_implementation import simulate_plate_main
            self.module = simulate_plate_main
            print("✓ Import successful (direct)")
            return self.module
        except ImportError as e:
            print(f"✗ Direct import failed: {e}")
        
        # Strategy 2: Import by file location
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "simulate_plate_main", 
                self.target_file
            )
            self.module = importlib.util.module_from_spec(spec)
            sys.modules['simulate_plate_main'] = self.module
            spec.loader.exec_module(self.module)
            print("✓ Import successful (by file location)")
            return self.module
        except Exception as e:
            print(f"✗ Import by location failed: {e}")
            traceback.print_exc()
            raise ImportError(f"Could not import simulation module from {self.target_file}")
    
    def get_main_function(self):
        """
        Identify the main entry point of the module
        """
        module = self.import_module()
        
        for func_name in ['main', 'run_simulation', 'run', 'execute']:
            if hasattr(module, func_name) and callable(getattr(module, func_name)):
                print(f"✓ Found entry point: {func_name}()")
                return getattr(module, func_name)
        
        print("✗ No standard entry point found")
        return None
    
    def profile_full_execution(self):
        """
        Profile complete execution with cProfile
        """
        with self.timer("Full Execution Profiling (cProfile)"):
            profiler = cProfile.Profile()
            main_func = self.get_main_function()
            
            if main_func is None:
                print("Cannot profile: no main function found")
                return
            
            try:
                profiler.enable()
                result = main_func()
                profiler.disable()
                
                self._save_profile_stats(
                    profiler, 
                    f"full_execution_{self.timestamp}"
                )
                
            except Exception as e:
                profiler.disable()
                print(f"Error during execution: {e}")
                traceback.print_exc()
    
    def profile_specific_functions(self):
        """
        Profile individual functions in the module
        """
        with self.timer("Specific Function Profiling"):
            module = self.import_module()
            
            # Get all public callable functions
            functions = [
                (name, getattr(module, name)) 
                for name in dir(module) 
                if not name.startswith('_') and callable(getattr(module, name))
            ]
            
            print(f"Found {len(functions)} public functions:")
            for name, _ in functions:
                print(f"  - {name}")
            
            # Profile only main entry points (avoid 'run' as it requires gmsh init)
            entry_points = ['main', 'run_simulation']
            for func_name, func in functions:
                if func_name not in entry_points:
                    continue
                
                print(f"\nProfiling {func_name}()...")
                try:
                    profiler = cProfile.Profile()
                    profiler.enable()
                    func()
                    profiler.disable()
                    
                    self._save_profile_stats(
                        profiler,
                        f"function_{func_name}_{self.timestamp}"
                    )
                except Exception as e:
                    print(f"  Error: {e}")
    
    def profile_line_by_line(self, function_names=None):
        """
        Perform line-by-line profiling on specific functions
        
        Args:
            function_names: List of function names to profile
        """
        if not LINE_PROFILER_AVAILABLE:
            print("\nLine profiling skipped: line_profiler not installed")
            print("Install with: pip install line_profiler")
            return
        
        with self.timer("Line-by-Line Profiling"):
            module = self.import_module()
            main_func = self.get_main_function()
            
            if main_func is None:
                print("Cannot profile: no main function found")
                return
            
            lp = LineProfiler()
            
            # Add functions to profile
            if function_names:
                for func_name in function_names:
                    if hasattr(module, func_name):
                        func = getattr(module, func_name)
                        lp.add_function(func)
                        print(f"  Added {func_name}() to line profiler")
                    else:
                        print(f"  Warning: {func_name}() not found")
            else:
                print("  No functions specified - profiling main only")
            
            # Add main function
            lp.add_function(main_func)
            
            # Run profiling
            try:
                lp_wrapper = lp(main_func)
                lp_wrapper()
                
                # Save results
                output_file = self.output_dir / f"line_profile_{self.timestamp}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    lp.print_stats(stream=f)
                print(f"✓ Saved to: {output_file}")
                
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
    
    def profile_memory(self):
        """
        Analyze memory usage during execution
        """
        if not MEMORY_PROFILER_AVAILABLE:
            print("\nMemory profiling skipped: memory_profiler not installed")
            print("Install with: pip install memory_profiler")
            return
        
        with self.timer("Memory Usage Analysis"):
            main_func = self.get_main_function()
            
            if main_func is None:
                print("Cannot profile: no main function found")
                return
            
            output_file = self.output_dir / f"memory_profile_{self.timestamp}.txt"
            
            # Redirect memory profiler output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            decorated_func = memory_profiler.profile(main_func, stream=f)
            
            try:
                with redirect_stdout(f):
                    decorated_func()
                
                # Save results
                with open(output_file, 'w', encoding='utf-8') as outf:
                    outf.write(f.getvalue())
                
                print(f"✓ Saved to: {output_file}")
                
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
    
    def benchmark_execution(self, iterations=3):
        """
        Run multiple iterations to benchmark execution time
        
        Args:
            iterations: Number of times to run the simulation
        """
        with self.timer(f"Performance Benchmark ({iterations} iterations)"):
            main_func = self.get_main_function()
            
            if main_func is None:
                print("Cannot benchmark: no main function found")
                return
            
            times = []
            for i in range(iterations):
                print(f"\nIteration {i+1}/{iterations}...")
                start = time.perf_counter()
                
                try:
                    main_func()
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    print(f"  Time: {elapsed:.3f} seconds")
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
            
            if times:
                print(f"\n{'─'*40}")
                print(f"Average: {sum(times)/len(times):.3f} seconds")
                print(f"Min:     {min(times):.3f} seconds")
                print(f"Max:     {max(times):.3f} seconds")
                print(f"Range:   {max(times)-min(times):.3f} seconds")
                
                # Save benchmark results
                output_file = self.output_dir / f"benchmark_{self.timestamp}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Benchmark Results ({iterations} iterations)\n")
                    f.write(f"{'='*50}\n\n")
                    for i, t in enumerate(times, 1):
                        f.write(f"Iteration {i}: {t:.3f} seconds\n")
                    f.write(f"\nAverage: {sum(times)/len(times):.3f} seconds\n")
                    f.write(f"Min:     {min(times):.3f} seconds\n")
                    f.write(f"Max:     {max(times):.3f} seconds\n")
                
                print(f"✓ Saved to: {output_file}")
    
    def _save_profile_stats(self, profiler, filename):
        """
        Save cProfile statistics to text and binary formats
        """
        base_path = self.output_dir / filename
        
        # Save human-readable text file
        txt_file = base_path.with_suffix('.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            f.write("=" * 80 + "\n")
            f.write(f"Profile Statistics: {filename}\n")
            f.write("=" * 80 + "\n\n")
            stats.print_stats(50)
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Top Time-Consuming Functions\n")
            f.write("=" * 80 + "\n")
            stats.sort_stats('time')
            stats.print_stats(20)
        
        # Save binary file for SnakeViz
        prof_file = base_path.with_suffix('.prof')
        profiler.dump_stats(str(prof_file))
        
        print(f"✓ Saved: {txt_file}")
        print(f"✓ Saved: {prof_file}")
    
    def print_summary(self):
        """
        Print summary and visualization instructions
        """
        print("\n" + "="*60)
        print("PROFILING COMPLETE")
        print("="*60)
        
        prof_files = list(self.output_dir.glob(f"*{self.timestamp}*.prof"))
        
        if prof_files:
            print("\n📊 Visualization with SnakeViz:")
            print("   pip install snakeviz")
            print(f"   snakeviz {prof_files[0]}")
        
        print(f"\n📁 All results saved to: {self.output_dir}/")
        print(f"   Timestamp: {self.timestamp}")
        
        print("\n📈 Files generated:")
        for file in sorted(self.output_dir.glob(f"*{self.timestamp}*")):
            print(f"   - {file.name}")


def main():
    """
    Main entry point for profiling
    """
    print("╔════════════════════════════════════════════════════════╗")
    print("║     Simulation Performance Profiling Suite             ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    profiler = SimulationProfiler()
    
    # Run profiling suite
    try:
        profiler.profile_full_execution()
        profiler.profile_specific_functions()
        
        # Optional: uncomment to enable
        # profiler.profile_line_by_line(['function1', 'function2'])
        # profiler.profile_memory()
        # profiler.benchmark_execution(iterations=3)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    profiler.print_summary()
    return 0


if __name__ == "__main__":
    exit(main())