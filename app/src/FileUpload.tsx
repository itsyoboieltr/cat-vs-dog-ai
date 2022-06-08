import {
  Box,
  CircularProgress,
  styled,
  SxProps,
  Theme,
  Typography,
  useTheme,
} from '@mui/material';
import { useEffect } from 'react';
import { Accept, useDropzone } from 'react-dropzone';
import * as tf from '@tensorflow/tfjs';

interface FileUploadProps {
  onUpload: (files: File[]) => void;
  onReset?: () => void;
  fileType?: Accept;
  fileCount?: number;
  sx?: SxProps<Theme>;
  progress: number;
  model: tf.LayersModel | undefined;
  loadModel: () => void;
}

const getColor = (
  theme: Theme,
  isDragAccept: boolean,
  isDragReject: boolean,
  isFileDialogActive: boolean
) => {
  if (isDragAccept) return '#00e676';
  if (isDragReject) return '#ff1744';
  if (isFileDialogActive) return theme.palette.primary.main;
  return theme.palette.text.disabled;
};

const FileUploadContainer = styled('div')(
  ({ color, disabled }: { color: string; disabled: boolean }) => ({
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: 20,
    borderWidth: 2,
    borderRadius: 2,
    borderColor: color,
    borderStyle: 'dashed',
    backgroundColor: 'transparent',
    color: color,
    outline: 'none',
    transition: 'border 0.24s ease-in-out, color 0.24s ease-in-out',
    userSelect: 'none',
    cursor: disabled ? 'auto' : 'pointer',
  })
);

export default function FileUpload(props: FileUploadProps) {
  const theme = useTheme();
  const disabled = props.model === undefined;
  const downloading = props.progress > 0 && props.model === undefined;
  const {
    acceptedFiles,
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragReject,
    isFileDialogActive,
  } = useDropzone({
    accept: props.fileType ?? undefined,
    maxFiles: props.fileCount ?? undefined,
    useFsAccessApi: false,
    disabled,
  });

  useEffect(() => {
    if (acceptedFiles.length) props.onUpload(acceptedFiles);
    else props.onReset?.();
  }, [acceptedFiles]);

  return (
    <FileUploadContainer
      {...getRootProps()}
      sx={props.sx}
      color={getColor(theme, isDragAccept, isDragReject, isFileDialogActive)}
      disabled={downloading}>
      <input {...getInputProps()} />
      <Box
        onClick={() => {
          if (disabled && !downloading) props.loadModel();
        }}
        sx={{
          textAlign: 'center',
          position: 'relative',
          top: '50%',
          transform: 'translateY(-50%)',
        }}>
        {downloading ? (
          <Box
            sx={{
              position: 'relative',
              display: 'inline-flex',
            }}>
            <CircularProgress variant={'determinate'} value={props.progress} />
            <Box
              sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
              <Typography
                variant='caption'
                component='div'
                color='text.secondary'>
                {`${Math.round(props.progress ?? 0)}%`}
              </Typography>
            </Box>
          </Box>
        ) : (
          <>
            {disabled ? (
              <Typography children={'Press to load model'} />
            ) : (
              <>
                <Typography
                  children={"Drag 'n' drop, press to select, or paste image"}
                />
                {acceptedFiles.map((file, index) => (
                  <Typography key={file.name + index} children={file.name} />
                ))}
              </>
            )}
          </>
        )}
      </Box>
    </FileUploadContainer>
  );
}
