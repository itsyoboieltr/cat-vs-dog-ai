import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Rank, Tensor } from '@tensorflow/tfjs';
import Grid from '@mui/material/Grid';
import { createTheme, ThemeProvider, Typography } from '@mui/material';
import FileUpload from './FileUpload';
import useEventListener from '@use-it/event-listener';
(window as any).global = window;

export default function App() {
  const theme = createTheme();
  const [model, setModel] = useState<tf.LayersModel>();
  const [progress, setProgress] = useState(0);
  const [prediction, setPrediction] = useState('');
  const [image, setImage] = useState('');

  async function loadModel() {
    const model = await tf.loadLayersModel('model/model.json', {
      onProgress: (progress) => setProgress(progress * 100),
    });
    setModel(model);
  }

  useEventListener('paste', async (e: ClipboardEvent) => {
    if (!e.clipboardData) return;
    for (const item of e.clipboardData.items) {
      if (item.type.indexOf('image') == -1) continue;
      const file = item.getAsFile();
      if (file) {
        const objectURL = URL.createObjectURL(file);
        await predict(objectURL);
        break;
      }
    }
  });

  const predict = async (objectURL: string) => {
    if (!model) return;
    setImage(objectURL);
    const image = new Image();
    image.src = objectURL;
    image.onload = async () => {
      const tensor = tf.browser.fromPixels(image);
      const resized = tf.image.resizeBilinear(tensor, [150, 150]);
      const normalized = resized.div(tf.scalar(255));
      const batched = normalized.expandDims(0);
      const predictions = model.predict(batched);
      const data = await (predictions as Tensor<Rank>).data();
      const prediction = data[0] > 0.5 ? 'Dog' : 'Cat';
      setPrediction(prediction);
    };
  };

  return (
    <ThemeProvider theme={theme}>
      <Grid container spacing={1} justifyContent={'center'}>
        <Grid item xs={12} sx={{ textAlign: 'center' }}>
          <Typography variant={'h5'} children={'Cat VS Dog AI'} />
          <Typography component={'div'} sx={{ fontWeight: 'lighter' }}>
            Author:{' '}
            <Typography
              children={'Norbert Elter'}
              onClick={() => window.open('https://github.com/itsyoboieltr')}
              sx={{
                fontWeight: 'lighter',
                display: 'inline',
                cursor: 'pointer',
                '&:hover': {
                  color: theme.palette.primary.main,
                  transition: theme.transitions.create('color', {
                    duration: theme.transitions.duration.shortest,
                  }),
                },
              }}
            />
          </Typography>
        </Grid>
        <Grid item xs={10} md={6}>
          <FileUpload
            sx={{ height: 150 }}
            fileCount={1}
            fileType={{ 'image/*': [] }}
            onUpload={async (files: File[]) => {
              const objectURL = URL.createObjectURL(files[0]);
              await predict(objectURL);
            }}
            progress={progress}
            model={model}
            loadModel={loadModel}
          />
        </Grid>
        <Grid item xs={12} />
        <Grid item xs={10} sm={6} md={4} lg={3} sx={{ textAlign: 'center' }}>
          {image && <img width={'100%'} src={image} draggable={false} />}
          {prediction && (
            <Typography fontWeight={'lighter'} children={prediction} />
          )}
        </Grid>
      </Grid>
    </ThemeProvider>
  );
}
