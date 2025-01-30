import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { HoverCard, HoverCardTrigger, HoverCardContent } from '@/components/ui/hover-card';
import { Info } from 'lucide-react';
import { debounce } from 'lodash';
import { scaleLinear } from 'd3-scale';

interface TrainingDataPoint {
  name: string;
  activations: number;
  weights: number;
  gradients: number;
}

interface TrainingVisualizationProps {
  data: TrainingDataPoint[];
}

const COLORS = {
  activations: '#3b82f6',
  weights: '#10b981',
  gradients: '#ef4444',
} as const;

const METRICS = Object.keys(COLORS) as (keyof typeof COLORS)[];

const METRIC_DESCRIPTIONS = {
  activations: "How strongly each layer responds to its input. Higher values mean the layer is more 'active' in processing the data.",
  weights: "The strength of connections between layers. Changes in weights show how the model is learning and adapting.",
  gradients: "How much each layer needs to change to improve. Larger values indicate areas where the model is actively learning.",
} as const;

// Downsample data for performance
const downsampleData = (data: TrainingDataPoint[], maxPoints: number): TrainingDataPoint[] => {
  if (data.length <= maxPoints) return data;
  const scale = scaleLinear().domain([0, maxPoints - 1]).range([0, data.length - 1]);
  return Array.from({ length: maxPoints }, (_, i) => data[Math.round(scale(i))]);
};

// Memoized smoothing function
const smoothData = (data: TrainingDataPoint[], smoothingFactor: number): TrainingDataPoint[] => {
  if (smoothingFactor === 0) return data;
  
  return data.reduce<TrainingDataPoint[]>((acc, point, index) => {
    if (index === 0) {
      acc.push(point);
      return acc;
    }

    const prevSmoothed = acc[index - 1];
    acc.push({
      name: point.name,
      activations: prevSmoothed.activations * (1 - smoothingFactor) + point.activations * smoothingFactor,
      weights: prevSmoothed.weights * (1 - smoothingFactor) + point.weights * smoothingFactor,
      gradients: prevSmoothed.gradients * (1 - smoothingFactor) + point.gradients * smoothingFactor,
    });
    return acc;
  }, []);
};

export function TrainingVisualization({ data }: TrainingVisualizationProps) {
  const [selectedMetrics, setSelectedMetrics] = useState<Set<string>>(new Set(['activations', 'weights']));
  const [smoothingFactor, setSmoothingFactor] = useState(0.3);
  const [autoScale, setAutoScale] = useState(true);

  // Downsample and smooth data
  const processedData = useMemo(() => 
    smoothData(downsampleData(data, 4000), smoothingFactor),
    [data, smoothingFactor]
  );

  // Memoize domain calculation for better performance
  const yDomain = useMemo(() => {
    if (!autoScale) return [0, 'auto'] as [number, 'auto'];
    
    const activeMetrics = Array.from(selectedMetrics);
    if (activeMetrics.length === 0) return ['auto', 'auto'] as ['auto', 'auto'];

    const values = processedData.flatMap(point => 
      activeMetrics.map(metric => Number(point[metric as keyof TrainingDataPoint]))
    ).filter((value): value is number => !isNaN(value));
    
    if (values.length === 0) return ['auto', 'auto'] as ['auto', 'auto'];
    
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    return [minValue * 0.9, maxValue * 1.1] as [number, number];
  }, [processedData, selectedMetrics, autoScale]);

  // Debounced smoothing factor update
  const debouncedSetSmoothingFactor = useCallback(
    debounce((value: number) => setSmoothingFactor(value), 100),
    []
  );

  const handleMetricToggle = useCallback((metric: string) => {
    setSelectedMetrics(prev => {
      const next = new Set(prev);
      if (next.has(metric)) {
        next.delete(metric);
      } else {
        next.add(metric);
      }
      return next;
    });
  }, []);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Neural Network Training Progress</CardTitle>
              <CardDescription className="mt-2">
                This visualization shows how your neural network evolves during training.
                Each line represents a different aspect of the network's behavior across its layers.
              </CardDescription>
            </div>
            <HoverCard>
              <HoverCardTrigger asChild>
                <button className="p-2 hover:bg-accent rounded-full">
                  <Info className="h-5 w-5 text-muted-foreground" />
                </button>
              </HoverCardTrigger>
              <HoverCardContent className="w-80">
                <div className="space-y-2">
                  <h4 className="font-semibold">Understanding the Visualization</h4>
                  <p className="text-sm text-muted-foreground">
                    The X-axis shows different layers in your neural network from input (left) to output (right).
                    The Y-axis shows the magnitude of different metrics that help us understand how the network is learning.
                  </p>
                </div>
              </HoverCardContent>
            </HoverCard>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="flex flex-col space-y-2">
              <div className="flex items-center space-x-2">
                <Label>Metrics to Display</Label>
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <button className="p-1 hover:bg-accent rounded-full">
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80">
                    <div className="space-y-2">
                      <h4 className="font-semibold">Available Metrics</h4>
                      {METRICS.map(metric => (
                        <div key={metric} className="space-y-1">
                          <p className="text-sm font-medium">{metric.charAt(0).toUpperCase() + metric.slice(1)}</p>
                          <p className="text-sm text-muted-foreground">
                            {METRIC_DESCRIPTIONS[metric as keyof typeof METRIC_DESCRIPTIONS]}
                          </p>
                        </div>
                      ))}
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </div>
              <div className="flex space-x-4">
                {METRICS.map(metric => (
                  <div key={metric} className="flex items-center space-x-2">
                    <Switch
                      checked={selectedMetrics.has(metric)}
                      onCheckedChange={() => handleMetricToggle(metric)}
                    />
                    <Label className="capitalize">{metric}</Label>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex flex-col space-y-2">
              <div className="flex items-center space-x-2">
                <Label>Smoothing Factor: {smoothingFactor.toFixed(2)}</Label>
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <button className="p-1 hover:bg-accent rounded-full">
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80">
                    <p className="text-sm text-muted-foreground">
                      Adjust this to make the lines smoother or more detailed.
                      Higher values (towards 1.0) make the graph smoother but may hide small variations.
                      Lower values (towards 0.0) show more detail but may be noisier.
                    </p>
                  </HoverCardContent>
                </HoverCard>
              </div>
              <Slider
                value={[smoothingFactor * 100]}
                onValueChange={([value]) => debouncedSetSmoothingFactor(value / 100)}
                min={0}
                max={100}
                step={1}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                checked={autoScale}
                onCheckedChange={setAutoScale}
              />
              <Label>Auto Scale Y-Axis</Label>
              <HoverCard>
                <HoverCardTrigger asChild>
                  <button className="p-1 hover:bg-accent rounded-full">
                    <Info className="h-4 w-4 text-muted-foreground" />
                  </button>
                </HoverCardTrigger>
                <HoverCardContent className="w-80">
                  <p className="text-sm text-muted-foreground">
                    When enabled, the graph will automatically adjust its scale to show all data points clearly.
                    When disabled, the scale starts from zero, which can help compare relative magnitudes.
                  </p>
                </HoverCardContent>
              </HoverCard>
            </div>

            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={processedData}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                  syncId="trainingChart"
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name"
                    label={{ value: 'Network Layers (Input → Output)', position: 'insideBottom', offset: -5 }}
                    interval={Math.floor(processedData.length / 10)}
                  />
                  <YAxis 
                    domain={yDomain as [number, number]}
                    label={{ value: 'Metric Values', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.9)' }}
                    formatter={(value: number, name: string) => [
                      value.toFixed(4),
                      name.charAt(0).toUpperCase() + name.slice(1)
                    ]}
                    labelFormatter={(label) => `Layer: ${label}`}
                  />
                  <Legend 
                    formatter={(value) => value.charAt(0).toUpperCase() + value.slice(1)}
                  />
                  {Array.from(selectedMetrics).map(metric => (
                    <Line
                      key={metric}
                      type="monotone"
                      dataKey={metric}
                      stroke={COLORS[metric as keyof typeof COLORS]}
                      dot={false}
                      name={metric}
                      isAnimationActive={false}
                      strokeDasharray={metric === 'weights' ? '5 5' : metric === 'gradients' ? '3 3' : undefined}
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 