'use client'
import React, { useState } from 'react';
import axios from 'axios';
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8001';

const DocumentSummarizer = ({ onSummaryComplete }) => {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setIsLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/summarize`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (typeof onSummaryComplete === 'function') {
        onSummaryComplete(response.data.summary);
      }
    } catch (error) {
      console.error('Error submitting file:', error);
      setError('An error occurred while summarizing the file. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-4">
          <Input 
            id="file-upload"
            type="file" 
            onChange={handleFileChange} 
            className="text-sm text-blue-700"
          />
        </div>
        <Button 
          type="submit" 
          className="bg-blue-500 text-white disabled:bg-gray-400 disabled:cursor-not-allowed"
          disabled={isLoading || !file}
        >
          {isLoading ? 'Summarizing...' : 'Summarize'}
        </Button>
      </form>
      {error && <p className="text-red-500 mb-4">{error}</p>}
      {isLoading && (
        <div className="mb-4 flex items-center">
          <svg className="animate-spin h-5 w-5 mr-3 text-blue-500" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Processing your file, please wait...
        </div>
      )}
    </div>
  );
};

export default DocumentSummarizer;