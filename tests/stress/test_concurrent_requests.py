"""
Stress tests for concurrent request handling
"""

import pytest
import torch
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, MagicMock, patch
import queue

from src.core.service import OptimizedFluxService
from src.api.server import create_app
from fastapi.testclient import TestClient


class TestConcurrentRequests:
    """Test system behavior under concurrent load"""
    
    @pytest.fixture
    def stress_service(self, mock_pipeline):
        """Create service for stress testing"""
        service = OptimizedFluxService()
        service.pipe = mock_pipeline
        
        # Make inference thread-safe
        service._lock = threading.Lock()
        
        def thread_safe_inference(*args, **kwargs):
            with service._lock:
                return service.inference(*args, **kwargs)
        
        service.thread_safe_inference = thread_safe_inference
        return service
    
    @pytest.mark.stress
    @pytest.mark.parametrize("num_threads", [2, 5, 10])
    def test_concurrent_inference_threads(self, stress_service, num_threads):
        """Test concurrent inference requests using threads"""
        results = queue.Queue()
        errors = queue.Queue()
        
        def run_inference(thread_id):
            try:
                result = stress_service.thread_safe_inference(
                    prompt=f"Thread {thread_id} test",
                    height=512,
                    width=512,
                    steps=5,
                    seed=thread_id
                )
                results.put((thread_id, result))
            except Exception as e:
                errors.put((thread_id, str(e)))
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            t = threading.Thread(target=run_inference, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=30)
        
        end_time = time.time()
        
        # Verify results
        assert errors.qsize() == 0, f"Errors occurred: {list(errors.queue)}"
        assert results.qsize() == num_threads
        
        # Check timing
        total_time = end_time - start_time
        print(f"Processed {num_threads} concurrent requests in {total_time:.2f}s")
        
        # Should handle concurrent requests efficiently
        assert total_time < num_threads * 2  # Should have some parallelism benefit
    
    @pytest.mark.stress
    def test_thread_pool_executor(self, stress_service):
        """Test using ThreadPoolExecutor for concurrent requests"""
        num_requests = 20
        max_workers = 4
        
        def process_request(request_id):
            return stress_service.thread_safe_inference(
                prompt=f"Request {request_id}",
                height=512,
                width=512,
                steps=5,
                seed=request_id
            )
        
        # Process requests in thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            start_time = time.time()
            futures = [executor.submit(process_request, i) for i in range(num_requests)]
            results = [f.result(timeout=10) for f in futures]
            end_time = time.time()
        
        # Verify all requests completed
        assert len(results) == num_requests
        assert all(r['inference_time'] > 0 for r in results)
        
        # Check performance
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests
        print(f"Average time per request: {avg_time_per_request:.3f}s")
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_memory_under_load(self, stress_service):
        """Test memory usage under concurrent load"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        num_concurrent = 5
        initial_memory = torch.cuda.memory_allocated()
        peak_memory = initial_memory
        
        def memory_tracking_inference(request_id):
            nonlocal peak_memory
            
            result = stress_service.thread_safe_inference(
                prompt=f"Memory test {request_id}",
                height=768,  # Larger size for memory stress
                width=768,
                steps=5
            )
            
            current_memory = torch.cuda.memory_allocated()
            peak_memory = max(peak_memory, current_memory)
            
            return result
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(memory_tracking_inference, i) 
                      for i in range(num_concurrent)]
            results = [f.result() for f in futures]
        
        # Calculate memory usage
        memory_increase_mb = (peak_memory - initial_memory) / (1024 * 1024)
        
        print(f"Peak memory increase: {memory_increase_mb:.2f} MB")
        
        # Memory should not grow excessively
        assert memory_increase_mb < 4000 * num_concurrent  # Reasonable limit
        
        # Cleanup
        torch.cuda.empty_cache()
    
    @pytest.mark.stress
    def test_api_concurrent_requests(self):
        """Test API server under concurrent load"""
        # Create test client with mock service
        mock_service = MagicMock()
        mock_service.inference.return_value = {
            'inference_time': 0.1,
            'processing_time': 0.01,
            'total_time': 0.11,
            'saved_files': ['test.png']
        }
        
        app = create_app(service=mock_service)
        client = TestClient(app)
        
        num_requests = 50
        results = queue.Queue()
        
        def make_api_request(request_id):
            try:
                response = client.post("/inference", json={
                    "prompt": f"API test {request_id}",
                    "width": 512,
                    "height": 512,
                    "steps": 5
                })
                results.put((request_id, response.status_code))
            except Exception as e:
                results.put((request_id, f"Error: {e}"))
        
        # Send concurrent requests
        threads = []
        start_time = time.time()
        
        for i in range(num_requests):
            t = threading.Thread(target=make_api_request, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        end_time = time.time()
        
        # Verify results
        successful_requests = sum(1 for _, status in results.queue if status == 200)
        assert successful_requests == num_requests
        
        # Check timing
        requests_per_second = num_requests / (end_time - start_time)
        print(f"Processed {requests_per_second:.2f} requests/second")
        
        # Should handle reasonable load
        assert requests_per_second > 10  # Should handle at least 10 req/s
    
    @pytest.mark.stress
    def test_queue_overflow_handling(self, stress_service):
        """Test behavior when request queue overflows"""
        # Simulate a queue with limited capacity
        request_queue = queue.Queue(maxsize=5)
        processing_times = []
        
        def process_with_queue(request_id):
            try:
                # Try to add to queue (non-blocking)
                request_queue.put_nowait(request_id)
                
                # Process request
                start = time.time()
                result = stress_service.thread_safe_inference(
                    prompt=f"Queue test {request_id}",
                    height=512,
                    width=512,
                    steps=5
                )
                processing_times.append(time.time() - start)
                
                # Remove from queue
                request_queue.get()
                return result
                
            except queue.Full:
                return {"error": "Queue full", "request_id": request_id}
        
        # Send more requests than queue can handle
        num_requests = 20
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_with_queue, i) 
                      for i in range(num_requests)]
            results = [f.result() for f in futures]
        
        # Some requests should be rejected due to queue overflow
        successful = sum(1 for r in results if "error" not in r)
        rejected = sum(1 for r in results if "error" in r and r["error"] == "Queue full")
        
        print(f"Successful: {successful}, Rejected: {rejected}")
        
        # Should handle overflow gracefully
        assert successful > 0
        assert successful + rejected == num_requests
    
    @pytest.mark.stress
    @pytest.mark.memory_intensive
    def test_sustained_load(self, stress_service):
        """Test system under sustained load over time"""
        duration_seconds = 10
        request_interval = 0.5  # 2 requests per second
        
        start_time = time.time()
        request_times = []
        errors = []
        
        request_id = 0
        while time.time() - start_time < duration_seconds:
            try:
                request_start = time.time()
                
                result = stress_service.thread_safe_inference(
                    prompt=f"Sustained load {request_id}",
                    height=512,
                    width=512,
                    steps=5
                )
                
                request_times.append(time.time() - request_start)
                request_id += 1
                
                # Wait for next request
                time.sleep(request_interval)
                
            except Exception as e:
                errors.append(str(e))
        
        # Analyze performance over time
        if request_times:
            avg_time = sum(request_times) / len(request_times)
            max_time = max(request_times)
            min_time = min(request_times)
            
            print(f"Processed {len(request_times)} requests over {duration_seconds}s")
            print(f"Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
            
            # Performance should remain stable
            assert max_time < avg_time * 2  # No extreme outliers
            assert len(errors) == 0  # No errors during sustained load
    
    @pytest.mark.stress
    def test_graceful_degradation(self, stress_service):
        """Test system degrades gracefully under extreme load"""
        # Simulate extreme load by limiting resources
        max_concurrent = 3
        semaphore = threading.Semaphore(max_concurrent)
        
        completed_requests = []
        rejected_requests = []
        
        def rate_limited_inference(request_id):
            acquired = semaphore.acquire(blocking=False)
            
            if acquired:
                try:
                    result = stress_service.thread_safe_inference(
                        prompt=f"Rate limited {request_id}",
                        height=512,
                        width=512,
                        steps=5
                    )
                    completed_requests.append(request_id)
                    return result
                finally:
                    semaphore.release()
            else:
                rejected_requests.append(request_id)
                return {"error": "Rate limited", "request_id": request_id}
        
        # Send many concurrent requests
        num_requests = 30
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(rate_limited_inference, i) 
                      for i in range(num_requests)]
            results = [f.result() for f in futures]
        
        print(f"Completed: {len(completed_requests)}, Rejected: {len(rejected_requests)}")
        
        # Should complete some requests and reject others gracefully
        assert len(completed_requests) > 0
        assert len(completed_requests) + len(rejected_requests) == num_requests
        
        # Rejected requests should have proper error messages
        rejected_results = [r for r in results if "error" in r]
        assert all(r["error"] == "Rate limited" for r in rejected_results)