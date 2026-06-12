#!/usr/bin/env python3
import subprocess
import json
import csv
import time
from datetime import datetime
import os

def check_iperf3():
    """Check if iperf3 is available in PATH"""
    try:
        result = subprocess.run(['which', 'iperf3'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"iperf3 found at: {result.stdout.strip()}")
            
            # Get version
            version = subprocess.run(['iperf3', '--version'], 
                                   capture_output=True, text=True)
            print(f"Version: {version.stdout.split()[2]}")
            return True
    except:
        pass
    
    print("iperf3 not found in PATH. Trying alternatives...")
    return False

def run_iperf_test(server_ip, duration=30, interval=1.0):
    """
    Run iperf3 test without sudo
    """
    print(f"\n{'='*60}")
    print(f"Testing bandwidth to {server_ip}")
    print(f"Duration: {duration} seconds")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Try different possible iperf3 locations
    iperf_paths = [
        'iperf3',
        '/usr/bin/iperf3',
        '/usr/local/bin/iperf3',
        f'{os.environ.get("HOME", "")}/.local/bin/iperf3'
    ]
    
    iperf_cmd = None
    for path in iperf_paths:
        if os.path.exists(path):
            iperf_cmd = path
            break
    
    if not iperf_cmd:
        print("ERROR: iperf3 not found. Please ask admin to install it.")
        print("Or use alternative methods below.")
        return None
    
    # Build command
    cmd = [
        iperf_cmd,
        '-c', server_ip,
        '-t', str(duration),
        '-i', str(interval),
        '-J',  # JSON output
        '--connect-timeout', '5000'  # 5 second connection timeout
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run test
        start_time = time.time()
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              timeout=duration + 10)
        
        elapsed = time.time() - start_time
        print(f"Test completed in {elapsed:.1f} seconds")
        
        if result.returncode != 0:
            print(f"\niperf3 returned error code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            return None
        
        # Parse JSON
        try:
            data = json.loads(result.stdout)
            return data
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            # Try to extract useful info from output anyway
            if "bits_per_second" in result.stdout:
                print("Found bandwidth data in output")
                # Extract using string methods
                lines = result.stdout.split('\n')
                for line in lines:
                    if "bits_per_second" in line:
                        print(line.strip())
            return None
            
    except subprocess.TimeoutExpired:
        print("Test timed out!")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def save_results(data, server_ip):
    """Save results to CSV"""
    if not data:
        return None
    
    # Create results directory in home folder
    results_dir = os.path.join(os.path.expanduser('~'), 'bandwidth_results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = os.path.join(results_dir, f'iperf_{server_ip}_{timestamp}.json')
    csv_file = os.path.join(results_dir, f'bandwidth_{server_ip}_{timestamp}.csv')
    
    # Save raw JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to: {json_file}")
    
    # Try to extract interval data
    results = []
    
    if 'intervals' in data:
        intervals = data['intervals']
        print(f"\nFound {len(intervals)} interval(s)")
        
        for interval in intervals:
            streams = interval.get('streams', [])
            for stream in streams:
                results.append({
                    'timestamp': interval.get('start', 0),
                    'bandwidth_mbps': stream.get('bits_per_second', 0) / 1_000_000,
                    'bytes': stream.get('bytes', 0),
                    'retransmits': stream.get('retransmits', 0) if 'retransmits' in stream else 0
                })
        
        # Save to CSV
        if results:
            with open(csv_file, 'w', newline='') as f:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"CSV data saved to: {csv_file}")
            
            # Print summary
            print_summary(results)
    
    # Also show end summary
    if 'end' in data:
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        end = data['end']
        if 'sum_received' in end:
            recv = end['sum_received']
            avg_bw = recv.get('bits_per_second', 0) / 1_000_000
            total_bytes = recv.get('bytes', 0)
            print(f"Average bandwidth: {avg_bw:.2f} Mbps")
            print(f"Total data received: {total_bytes / 1_000_000:.2f} MB")
            print(f"Duration: {recv.get('seconds', 0):.1f} seconds")
    
    return csv_file

def print_summary(results):
    """Print summary statistics"""
    if not results:
        return
    
    bandwidths = [r['bandwidth_mbps'] for r in results]
    
    print("\n" + "="*60)
    print("INTERVAL SUMMARY")
    print("="*60)
    print(f"Number of samples: {len(bandwidths)}")
    print(f"Time range: {results[0]['timestamp']:.1f}s to {results[-1]['timestamp']:.1f}s")
    
    if bandwidths:
        avg = sum(bandwidths) / len(bandwidths)
        max_bw = max(bandwidths)
        min_bw = min(bandwidths)
        
        print(f"Average bandwidth: {avg:.2f} Mbps")
        print(f"Maximum bandwidth: {max_bw:.2f} Mbps")
        print(f"Minimum bandwidth: {min_bw:.2f} Mbps")
        
        # Simple text-based visualization
        print("\nBandwidth trend (simplified):")
        for i, bw in enumerate(bandwidths[:20]):  # Show first 20 samples
            bar_length = int((bw / max_bw) * 50) if max_bw > 0 else 0
            bar = '█' * bar_length
            print(f"{i*0.5:5.1f}s: {bar:50} {bw:6.1f} Mbps")

def main():
    """Main function"""
    server_ip = "131.230.191.156"
    
    print("Bandwidth Measurement (No Sudo Required)")
    print("="*60)
    
    # First, check if iperf3 is available
    if not check_iperf3():
        print("\nTrying alternative measurement methods...")
        # Try alternative methods
        try_alternative_methods(server_ip)
        return
    
    # Ask for test duration
    print("\nTest options:")
    print("1. Quick test (10 seconds)")
    print("2. Standard test (60 seconds)")
    print("3. Full test (300 seconds = 5 minutes)")
    print("4. Custom duration")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        duration = 10
        interval = 0.5
    elif choice == '2':
        duration = 60
        interval = 1.0
    elif choice == '3':
        duration = 300
        interval = 1.0
    elif choice == '4':
        try:
            duration = int(input("Enter duration in seconds: "))
            interval = float(input("Enter reporting interval (0.5-5.0): "))
        except:
            print("Invalid input, using defaults")
            duration = 60
            interval = 1.0
    else:
        duration = 30
        interval = 1.0
    
    print(f"\nWill run {duration} second test with {interval} second intervals")
    
    # Run test
    data = run_iperf_test(server_ip, duration, interval)
    
    if data:
        save_results(data, server_ip)
    else:
        print("\nTest failed. Possible reasons:")
        print("1. iperf3 server not running on target")
        print("2. Network connectivity issues")
        print("3. Firewall blocking port 5201")
        print("\nTry these troubleshooting steps:")
        print(f"  ping {server_ip}")
        print(f"  nc -zv {server_ip} 5201")
        print(f"  telnet {server_ip} 5201")

def try_alternative_methods(server_ip):
    """Try methods that don't require iperf3"""
    print("\n" + "="*60)
    print("Alternative Measurement Methods")
    print("="*60)
    
    print("\nMethod 1: Using Python sockets (requires server cooperation)")
    
    # Simple ping test first
    print("\nChecking basic connectivity...")
    try:
        result = subprocess.run(['ping', '-c', '3', server_ip],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {server_ip} is reachable via ping")
        else:
            print(f"✗ Cannot ping {server_ip}")
    except:
        print("Ping test failed")
    
    # Try netcat if available
    print("\nChecking port 5201...")
    try:
        result = subprocess.run(['nc', '-z', '-w', '2', server_ip, '5201'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Port 5201 appears to be open")
            print("\nYou can manually run iperf3:")
            print(f"  iperf3 -c {server_ip} -t 300 -i 1")
        else:
            print("✗ Port 5201 is not open or not accessible")
    except FileNotFoundError:
        print("netcat not available")
    
    print("\nWithout iperf3, you can use:")
    print("1. Ask admin to install iperf3 system-wide")
    print("2. Use web-based speed test if available")
    print("3. Use scp/wget for approximate measurements")

if __name__ == "__main__":
    main()