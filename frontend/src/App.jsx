import { useState } from 'react';
import { Container, Typography, Box, CircularProgress, Alert, Slider } from '@mui/material';
import ImageUploader from './components/ImageUploader';
import ResultsGrid from './components/ResultsGrid';

function App() {
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [similarityThreshold, setSimilarityThreshold] = useState(0);

    const handleSearch = async ({ imageFile, imageUrl }) => {
        setIsLoading(true);
        setError(null);
        setResults([]);
        const formData = new FormData();
        if (imageFile) formData.append("image_file", imageFile);
        else if (imageUrl) formData.append("image_url", imageUrl);
        else {
            setError("Please provide an image or a URL.");
            setIsLoading(false);
            return;
        }

        try {
            const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/search`, {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            if (data.status === 'success') {
                setResults(data.data);
            } else {
                setError(data.message || "An unknown error occurred.");
            }
        } catch (err) {
            setError("Failed to connect to the server. Please ensure the backend is running.");
        } finally {
            setIsLoading(false);
        }
    };

    const filteredResults = results.filter(product => {
        const similarity = (1 / (1 + product.similarity_score)) * 100;
        return similarity >= similarityThreshold;
    });

    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
                <Typography variant="h3" component="h1" fontWeight="bold" gutterBottom>
                    Lookalike Lens
                </Typography>
                <Typography variant="h6" color="text.secondary">
                    Find visually similar products in seconds.
                </Typography>
            </Box>

            <ImageUploader onSearch={handleSearch} isLoading={isLoading} />

            <Box sx={{ mt: 4 }}>
                {isLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                        <CircularProgress />
                    </Box>
                )}
                {error && <Alert severity="error" sx={{ my: 2 }}>{error}</Alert>}
                {results.length > 0 && (
                    <Box>
                         <Box sx={{ maxWidth: 'md', mx: 'auto', my: 4 }}>
                            <Typography gutterBottom>
                                Similarity Threshold: {similarityThreshold}%
                            </Typography>
                            <Slider
                                value={similarityThreshold}
                                onChange={(e, newValue) => setSimilarityThreshold(newValue)}
                                aria-labelledby="similarity-slider"
                                valueLabelDisplay="auto"
                                step={1}
                                marks
                                min={0}
                                max={100}
                            />
                        </Box>
                        {filteredResults.length > 0 ? (
                            <ResultsGrid results={filteredResults} />
                        ) : (
                            <Typography align="center" color="text.secondary">No results match the selected filter.</Typography>
                        )}
                    </Box>
                )}
            </Box>
            <Typography align="center" color="text.secondary">Have a good one!</Typography>
        </Container>
    );
}

export default App;
