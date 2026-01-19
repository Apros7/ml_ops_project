# Performance Testing with Locust

This directory contains load testing scripts for the license plate recognition API.

## Prerequisites

Install Locust:
```bash
uv add locust
```

## Running Locust Locally

### Interactive Mode (Recommended for Testing)

Start Locust in interactive mode:
```bash
locust -f locustfile.py --host http://localhost:8000
```

Then open your browser to `http://localhost:8089` to access the Locust web UI.

### Headless Mode (For CI/CD)

Run Locust in headless mode with specified parameters:
```bash
locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 60s --html report.html
```

Parameters:
- `-u 50`: Simulate 50 concurrent users
- `-r 10`: Spawn 10 users per second
- `-t 60s`: Run for 60 seconds
- `--html report.html`: Generate HTML report

## What Metrics to Look For

### Performance Targets
- **Average response time**: < 2 seconds
- **95th percentile response time**: < 5 seconds
- **Failure rate**: < 1%
- **Requests per second**: Monitor throughput

### Key Metrics

1. **Response Times**
   - Min, Max, Average
   - Median (50th percentile)
   - 95th and 99th percentiles

2. **Request Rates**
   - Total requests per second (RPS)
   - Successful vs failed requests

3. **Error Rates**
   - HTTP error codes (400, 500, etc.)
   - Timeout errors
   - Connection errors

4. **Concurrent Users**
   - Number of simulated users
   - Active connections

## Interpreting Results

- **Green (< 2s)**: Good performance
- **Yellow (2-5s)**: Acceptable, monitor closely
- **Red (> 5s)**: Performance issues, investigate

If failure rate exceeds 1%, check:
- API server logs
- Resource utilization (CPU, memory)
- Network latency
- Model loading times
