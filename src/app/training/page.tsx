'use client';

import React, { useState, useCallback } from 'react';
import { TrainingVisualization } from '@/components/TrainingVisualization';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';

interface TrainingDataPoint {
  name: string;
  activations: number;
  weights: number;
  gradients: number;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export default function TrainingPage() {
  const [data, setData] = useState<TrainingDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const loadTrainingData = useCallback(async (file: File) => {
    if (file.size > MAX_FILE_SIZE) {
      toast({
        title: 'Error',
        description: 'File size exceeds 10MB limit',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    try {
      const text = await file.text();
      const jsonData = JSON.parse(text);
      
      // Validate data structure
      if (!Array.isArray(jsonData)) {
        throw new Error('Invalid data format: expected an array');
      }

      // Transform and validate in a single pass
      const transformedData = jsonData.map((item, index) => {
        if (!item || typeof item !== 'object') {
          throw new Error(`Invalid item at index ${index}: expected an object`);
        }

        const { name, activations, weights, gradients } = item;
        
        if (typeof activations !== 'number' || 
            typeof weights !== 'number' || 
            typeof gradients !== 'number') {
          throw new Error(`Invalid metrics at index ${index}: expected numbers`);
        }

        return { name: name || `Layer ${index}`, activations, weights, gradients };
      });

      setData(transformedData);
      toast({
        title: 'Success',
        description: `Loaded ${transformedData.length} data points successfully`,
      });
    } catch (error) {
      console.error('Error loading training data:', error);
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to load training data',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.json')) {
      loadTrainingData(file);
    } else {
      toast({
        title: 'Error',
        description: 'Please drop a valid JSON file',
        variant: 'destructive',
      });
    }
  }, [loadTrainingData, toast]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.json')) {
      loadTrainingData(file);
    }
  }, [loadTrainingData]);

  return (
    <div className="container mx-auto p-4 space-y-4">
      <h1 className="text-2xl font-bold">Training Visualization</h1>
      
      {data.length === 0 ? (
        <Card
          className="border-dashed border-2 p-8 text-center cursor-pointer transition-colors hover:border-primary"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
        >
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Drag and drop your neuroscope_training.json file here, or
            </p>
            <Button
              variant="outline"
              onClick={() => document.getElementById('file-upload')?.click()}
              disabled={isLoading}
            >
              {isLoading ? 'Loading...' : 'Choose File'}
            </Button>
            <input
              id="file-upload"
              type="file"
              accept=".json"
              className="hidden"
              onChange={handleFileSelect}
              disabled={isLoading}
            />
          </div>
        </Card>
      ) : (
        <>
          <Button 
            variant="outline" 
            onClick={() => setData([])}
            className="mb-4"
          >
            Load Different File
          </Button>
          <TrainingVisualization data={data} />
        </>
      )}
    </div>
  );
} 