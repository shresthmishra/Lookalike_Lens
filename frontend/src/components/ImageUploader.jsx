import { useState } from 'react';
import { Box, Button, TextField, Typography, Paper, Stack } from '@mui/material';

export default function ImageUploader({ onSearch, isLoading }) {
    const [imageFile, setImageFile] = useState(null);
    const [imageUrl, setImageUrl] = useState('');
    const [previewUrl, setPreviewUrl] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImageFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setImageUrl('');
        }
    };

    const handleUrlChange = (e) => {
        setImageUrl(e.target.value);
        setImageFile(null);
        setPreviewUrl(null);
    };

    const handleSearch = () => {
        if (imageUrl) setPreviewUrl(imageUrl);
        onSearch({ imageFile, imageUrl });
    };

    return (
        <Paper elevation={3} sx={{ p: 4, maxWidth: 'md', mx: 'auto' }}>
            <Stack spacing={2}>
                <Typography variant="h6" component="h2" gutterBottom>
                    Find a Product
                </Typography>
                <TextField
                    label="Paste an image URL"
                    variant="outlined"
                    fullWidth
                    value={imageUrl}
                    onChange={handleUrlChange}
                    disabled={isLoading}
                />
                <Typography align="center" color="text.secondary">OR</Typography>
                <Button variant="outlined" component="label" disabled={isLoading}>
                    Upload an Image
                    <input type="file" hidden accept="image/*" onChange={handleFileChange} />
                </Button>
                {previewUrl && (
                    <Box sx={{ mt: 2, border: '1px solid lightgray', borderRadius: 1 }}>
                        <img src={previewUrl} alt="Preview" style={{ width: '100%', height: 'auto', display: 'block' }} />
                    </Box>
                )}
                <Button
                    variant="contained"
                    size="large"
                    onClick={handleSearch}
                    disabled={(!imageFile && !imageUrl) || isLoading}
                    sx={{ mt: 2 }}
                >
                    Find Lookalikes
                </Button>
            </Stack>
        </Paper>
    );
}