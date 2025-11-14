import pstats
from pathlib import Path
import sys
from datetime import datetime

# Configuration du répertoire des résultats
PROFILING_RESULTS_DIR = Path("backend/performance/profiling_results")

def save_analysis_to_file(profile_file, output_file):
    """
    Analyze profiling results and save to text file
    """
    print(f"📊 Analyzing {profile_file.name}...")
    
    stats = pstats.Stats(str(profile_file))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"PROFILE ANALYSIS: {profile_file.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 1. Top 20 functions by cumulative time
        f.write("="*80 + "\n")
        f.write("TOP 20 FUNCTIONS BY CUMULATIVE TIME\n")
        f.write("="*80 + "\n")
        f.write("(Total time spent in function and all functions it calls)\n\n")
        
        # Capture cumulative time stats
        stream = open('temp_cumulative.txt', 'w+', encoding='utf-8')
        stats_stream = pstats.Stats(str(profile_file), stream=stream)
        stats_stream.sort_stats('cumulative')
        stats_stream.print_stats(20)
        stream.seek(0)
        f.write(stream.read())
        stream.close()
        Path('temp_cumulative.txt').unlink()
        
        # 2. Top 20 functions by internal time
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 FUNCTIONS BY INTERNAL TIME\n")
        f.write("="*80 + "\n")
        f.write("(Time spent in function itself, excluding calls)\n\n")
        
        stream = open('temp_internal.txt', 'w+', encoding='utf-8')
        stats_stream = pstats.Stats(str(profile_file), stream=stream)
        stats_stream.sort_stats('time')
        stats_stream.print_stats(20)
        stream.seek(0)
        f.write(stream.read())
        stream.close()
        Path('temp_internal.txt').unlink()
        
        # 3. Top 20 most called functions
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 MOST CALLED FUNCTIONS\n")
        f.write("="*80 + "\n\n")
        
        stream = open('temp_calls.txt', 'w+', encoding='utf-8')
        stats_stream = pstats.Stats(str(profile_file), stream=stream)
        stats_stream.sort_stats('calls')
        stats_stream.print_stats(20)
        stream.seek(0)
        f.write(stream.read())
        stream.close()
        Path('temp_calls.txt').unlink()
        
        # 4. Callers for top functions
        f.write("\n" + "="*80 + "\n")
        f.write("CALL RELATIONSHIPS (Top 5 functions)\n")
        f.write("="*80 + "\n\n")
        
        stream = open('temp_callers.txt', 'w+', encoding='utf-8')
        stats_stream = pstats.Stats(str(profile_file), stream=stream)
        stats_stream.sort_stats('cumulative')
        stats_stream.print_callers(5)
        stream.seek(0)
        f.write(stream.read())
        stream.close()
        Path('temp_callers.txt').unlink()
        
        # 5. Performance summary
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n")
        
        stats.sort_stats('cumulative')
        
        # Get total time
        total_time = 0
        for func_stat in stats.stats.values():
            total_time += func_stat[3]  # cumulative time
        
        f.write(f"Total execution time: {total_time:.2f} seconds\n")
        f.write(f"Total function calls: {stats.total_calls}\n")
        f.write(f"Primitive calls: {stats.prim_calls}\n")
        
        # Find bottlenecks (functions taking >5% of total time)
        f.write("\n🔥 BOTTLENECKS (functions taking >5% of total time):\n")
        bottlenecks = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if ct > total_time * 0.05:
                percentage = (ct / total_time) * 100
                func_name = f"{func[0]}:{func[1]}({func[2]})"
                bottlenecks.append((percentage, ct, nc, func_name))
        
        bottlenecks.sort(reverse=True)
        for i, (pct, time, calls, name) in enumerate(bottlenecks[:10], 1):
            f.write(f"  {i}. {pct:5.1f}% - {time:7.2f}s - {calls:6d} calls - {name}\n")
        
        # 6. File-specific statistics
        f.write("\n" + "="*80 + "\n")
        f.write("FILE-SPECIFIC STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # Group by file
        file_stats = {}
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename = func[0]
            if filename not in file_stats:
                file_stats[filename] = {'time': 0, 'calls': 0, 'functions': 0}
            file_stats[filename]['time'] += ct
            file_stats[filename]['calls'] += nc
            file_stats[filename]['functions'] += 1
        
        # Sort files by total time
        sorted_files = sorted(file_stats.items(), key=lambda x: x[1]['time'], reverse=True)
        
        f.write("Top files by cumulative time:\n")
        for i, (filename, data) in enumerate(sorted_files[:10], 1):
            pct = (data['time'] / total_time) * 100
            f.write(f"  {i}. {pct:5.1f}% - {data['time']:7.2f}s - {data['calls']:6d} calls - {data['functions']:3d} funcs - {filename}\n")

def analyze_profile(profile_file):
    """
    Analyze and display profiling results in various formats
    """
    print("="*80)
    print(f"PROFILE ANALYSIS: {profile_file.name}")
    print("="*80)
    
    stats = pstats.Stats(str(profile_file))
    
    # 1. Top 20 functions by cumulative time
    print("\n" + "="*80)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    print("(Total time spent in function and all functions it calls)")
    print()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    # 2. Top 20 functions by internal time
    print("\n" + "="*80)
    print("TOP 20 FUNCTIONS BY INTERNAL TIME")
    print("="*80)
    print("(Time spent in function itself, excluding calls)")
    print()
    stats.sort_stats('time')
    stats.print_stats(20)
    
    # 3. Top 20 most called functions
    print("\n" + "="*80)
    print("TOP 20 MOST CALLED FUNCTIONS")
    print("="*80)
    stats.sort_stats('calls')
    stats.print_stats(20)
    
    # 4. Callers and callees for top functions
    print("\n" + "="*80)
    print("CALL RELATIONSHIPS (Top 5 functions)")
    print("="*80)
    stats.sort_stats('cumulative')
    stats.print_callers(5)
    
    # 5. Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    stats.sort_stats('cumulative')
    
    # Get total time
    total_time = 0
    for func_stat in stats.stats.values():
        total_time += func_stat[3]  # cumulative time
    
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total function calls: {stats.total_calls}")
    print(f"Primitive calls: {stats.prim_calls}")
    
    # Find bottlenecks (functions taking >5% of total time)
    print("\n🔥 BOTTLENECKS (functions taking >5% of total time):")
    bottlenecks = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        if ct > total_time * 0.05:
            percentage = (ct / total_time) * 100
            func_name = f"{func[0]}:{func[1]}({func[2]})"
            bottlenecks.append((percentage, ct, nc, func_name))
    
    bottlenecks.sort(reverse=True)
    for i, (pct, time, calls, name) in enumerate(bottlenecks[:10], 1):
        print(f"  {i}. {pct:5.1f}% - {time:7.2f}s - {calls:6d} calls - {name}")

def compare_profiles(profile_files):
    """
    Compare multiple profile files to see differences
    """
    if len(profile_files) < 2:
        print("Need at least 2 profile files to compare")
        return
    
    print("="*80)
    print("PROFILE COMPARISON")
    print("="*80)
    
    all_stats = []
    for pf in profile_files:
        stats = pstats.Stats(str(pf))
        all_stats.append((pf.name, stats))
        print(f"\n📁 {pf.name}")
        print(f"   Total calls: {stats.total_calls}")

def find_latest_profile(directory=PROFILING_RESULTS_DIR):
    """
    Find the most recent profile file in profiling_results directory
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return None
    
    prof_files = sorted(directory.glob("*.prof"), 
                       key=lambda x: x.stat().st_mtime,
                       reverse=True)
    
    if not prof_files:
        print(f"❌ No profile files found in {directory}")
        return None
    
    return prof_files[0]

def find_all_profiles(directory=PROFILING_RESULTS_DIR):
    """
    Find all profile files in profiling_results directory
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    prof_files = sorted(directory.glob("*.prof"), 
                       key=lambda x: x.stat().st_mtime,
                       reverse=True)
    return prof_files

def interactive_analysis():
    """
    Interactive analysis menu with save to file option
    """
    print("╔════════════════════════════════════════════════════════╗")
    print("║        Profile Results Analysis Tool                   ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Find all profile files in profiling_results
    prof_files = find_all_profiles()
    
    if not prof_files:
        print(f"\n❌ No profile files found in {PROFILING_RESULTS_DIR}/")
        print("   Run profile_simulate_plate.py first to generate profile files")
        return
    
    print(f"\n📁 Found {len(prof_files)} profile file(s) in profiling_results:\n")
    for i, pf in enumerate(prof_files, 1):
        size_kb = pf.stat().st_size / 1024
        mtime = datetime.fromtimestamp(pf.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        print(f"  {i}. {pf.name} ({size_kb:.1f} KB, {mtime})")
    
    print("\n" + "─"*60)
    print("Options:")
    print("  1. Analyze latest profile (console only)")
    print("  2. Analyze specific profile (console only)")
    print("  3. Save analysis to file (latest profile)")
    print("  4. Save analysis to file (specific profile)")
    print("  5. Compare two profiles")
    print("  6. Batch analyze all profiles")
    print("  7. Exit")
    print("─"*60)
    
    try:
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            analyze_profile(prof_files[0])
        
        elif choice == '2':
            idx = int(input(f"Enter profile number (1-{len(prof_files)}): ")) - 1
            if 0 <= idx < len(prof_files):
                analyze_profile(prof_files[idx])
            else:
                print("Invalid selection")
        
        elif choice == '3':
            output_file = prof_files[0].stem + "_analysis.txt"
            output_path = PROFILING_RESULTS_DIR / output_file
            save_analysis_to_file(prof_files[0], output_path)
            print(f"✅ Analysis saved to: {output_path}")
        
        elif choice == '4':
            idx = int(input(f"Enter profile number (1-{len(prof_files)}): ")) - 1
            if 0 <= idx < len(prof_files):
                output_file = prof_files[idx].stem + "_analysis.txt"
                output_path = PROFILING_RESULTS_DIR / output_file
                save_analysis_to_file(prof_files[idx], output_path)
                print(f"✅ Analysis saved to: {output_path}")
            else:
                print("Invalid selection")
        
        elif choice == '5':
            idx1 = int(input(f"Enter first profile number (1-{len(prof_files)}): ")) - 1
            idx2 = int(input(f"Enter second profile number (1-{len(prof_files)}): ")) - 1
            if 0 <= idx1 < len(prof_files) and 0 <= idx2 < len(prof_files):
                compare_profiles([prof_files[idx1], prof_files[idx2]])
            else:
                print("Invalid selection")
        
        elif choice == '6':
            batch_analyze_all_profiles()
        
        elif choice == '7':
            print("Goodbye!")
        
        else:
            print("Invalid choice")
    
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")

def batch_analyze_all_profiles():
    """
    Analyze all profile files and save results to text files
    """
    prof_files = find_all_profiles()
    
    if not prof_files:
        print(f"❌ No profile files found in {PROFILING_RESULTS_DIR}/")
        return
    
    print(f"🔍 Analyzing {len(prof_files)} profile files...")
    
    for profile_file in prof_files:
        output_file = profile_file.stem + "_analysis.txt"
        output_path = PROFILING_RESULTS_DIR / output_file
        
        try:
            save_analysis_to_file(profile_file, output_path)
            print(f"✅ Saved: {output_file}")
        except Exception as e:
            print(f"❌ Error analyzing {profile_file.name}: {e}")

def main():
    """
    Main entry point
    """
    # Create profiling results directory if it doesn't exist
    PROFILING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch analyze all profiles
            batch_analyze_all_profiles()
        elif sys.argv[1] == "--save":
            # Save analysis of latest profile
            latest_profile = find_latest_profile()
            if latest_profile:
                output_file = latest_profile.stem + "_analysis.txt"
                output_path = PROFILING_RESULTS_DIR / output_file
                save_analysis_to_file(latest_profile, output_path)
                print(f"✅ Analysis saved to: {output_path}")
            else:
                print("❌ No profile files found")
        elif sys.argv[1] == "--dir":
            # Custom directory
            custom_dir = Path(sys.argv[2])
            prof_files = find_all_profiles(custom_dir)
            if prof_files:
                for pf in prof_files:
                    output_file = pf.stem + "_analysis.txt"
                    output_path = custom_dir / output_file
                    save_analysis_to_file(pf, output_path)
                    print(f"✅ Saved: {output_path}")
        else:
            # Analyze specific file from command line
            profile_file = Path(sys.argv[1])
            if not profile_file.exists():
                # Try to find in profiling_results directory
                profile_file = PROFILING_RESULTS_DIR / sys.argv[1]
            
            if profile_file.exists():
                if len(sys.argv) > 2 and sys.argv[2] == "--save":
                    output_file = profile_file.stem + "_analysis.txt"
                    output_path = PROFILING_RESULTS_DIR / output_file
                    save_analysis_to_file(profile_file, output_path)
                    print(f"✅ Analysis saved to: {output_path}")
                else:
                    analyze_profile(profile_file)
            else:
                print(f"❌ File not found: {sys.argv[1]}")
                print(f"   Also checked in: {PROFILING_RESULTS_DIR}/")
    else:
        # Interactive mode
        interactive_analysis()

if __name__ == "__main__":
    print("Profile Analysis Tool")
    print(f"Looking for profiles in: {PROFILING_RESULTS_DIR}/")
    print("\nUsage:")
    print("  python script.py                    # Interactive mode")
    print("  python script.py profile.prof       # Analyze specific profile")
    print("  python script.py profile.prof --save # Save analysis to file")
    print("  python script.py --batch            # Analyze all profiles")
    print("  python script.py --save             # Save latest profile analysis")
    print("  python script.py --dir /path/to/profiles # Analyze custom directory")
    print()
    main()