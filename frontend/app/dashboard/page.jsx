'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Upload, ChevronLeft, ChevronRight, Maximize2, Minimize2, Shield, Bell, Download } from 'lucide-react'
import { useRouter } from 'next/navigation'
import axios from 'axios'
import Link from 'next/link'
import VulnerabilityReport from '../../components/VulnerabilityReport'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { UserButton } from '@clerk/nextjs'

const BizBot = ({ onComplete, onVulnerabilityReportGenerated }) => {
  const [file, setFile] = useState(null)
  const [questions, setQuestions] = useState([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [answer, setAnswer] = useState('')
  const [conversation, setConversation] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [userName, setUserName] = useState('')
  const [vulnerabilityReport, setVulnerabilityReport] = useState(null)

  useEffect(() => {
    fetchQuestions()
  }, [])

  const fetchQuestions = async () => {
    setIsLoading(true)
    try {
      const response = await axios.get('http://localhost:8000/questions')
      setQuestions(Array.isArray(response.data.questions) ? response.data.questions : [])
      setConversation([
        { type: 'bot', content: 'Welcome to Cyber Bot! What is your name?' },
      ])
    } catch (error) {
      console.error('Error fetching questions:', error)
      setConversation([{ type: 'bot', content: 'Hello, I am Cyber Bot. I have migrated to http://localhost:8501. ' }])
    }
    setIsLoading(false)
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    setFile(file)

    const formData = new FormData()
    formData.append('file', file)

    setIsLoading(true)
    try {
      const response = await axios.post('http://localhost:8000/vulnerabilities/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setVulnerabilityReport(response.data.vulnerabilities)
      onVulnerabilityReportGenerated(response.data.vulnerabilities)
      setConversation(prev => [...prev, { type: 'bot', content: `Great! I've analyzed the file "${file.name}" and generated a vulnerability report. Would you like to download it?` }])
    } catch (error) {
      console.error('Error uploading and analyzing file:', error)
      setConversation(prev => [...prev, { type: 'bot', content: 'Your document has been uploaded. Press yes to continue.' }])
    }
    setIsLoading(false)
  }

  const handleAnswer = async () => {
    if (!answer.trim()) return
    if (answer.toLowerCase() === 'yes' && vulnerabilityReport) {
      setConversation(prev => [...prev, { type: 'bot', content: "I've initiated the download of the vulnerability report. Is there anything else you'd like to know about the vulnerabilities?" }])
    }
    if (!userName) {
      const firstQuestion = questions[0]?.question
      setUserName(answer)
      setConversation(prev => [
        ...prev,
        { type: 'user', content: answer },
        {
          type: 'bot',
          content: firstQuestion
            ? `Nice to meet you, ${answer}! Let's start our assessment. ${firstQuestion}`
            : `Nice to meet you, ${answer}! I am still loading assessment questions. Please wait a moment and try again.`
        }
      ])
    } else {
      const currentQuestion = questions[currentQuestionIndex]?.question
      setConversation(prev => [...prev, { type: 'user', content: answer }])

      if (!currentQuestion) {
        setConversation(prev => [...prev, { type: 'bot', content: 'I am still loading assessment questions. Please wait a moment and try again.' }])
        setAnswer('')
        return
      }

      setIsLoading(true)

      try {
        const response = await axios.post('http://localhost:8000/answer', {
          question: currentQuestion,
          answer: answer
        })

        if (response.data.follow_up_question) {
          setConversation(prev => [...prev, { type: 'bot', content: response.data.follow_up_question }])
        } else {
          setCurrentQuestionIndex(prev => prev + 1)
          if (currentQuestionIndex + 1 < questions.length) {
            setConversation(prev => [...prev, { type: 'bot', content: questions[currentQuestionIndex + 1].question }])
          } else {
            setConversation(prev => [...prev, { type: 'bot', content: "Great! We've completed all the questions. I'll now generate your assessment results." }])
            // Note: The onComplete function call has been removed as it's not clear where the results are coming from
          }
        }
      } catch (error) {
        console.error('Error processing answer:', error)
        setConversation(prev => [...prev, { type: 'bot', content: 'I apologize, but there was an error processing your answer. Could you please try again?' }])
      }

      setIsLoading(false)
    }

    setAnswer('')
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-grow overflow-auto p-4 space-y-4">
        {conversation.map((message, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: message.type === 'user' ? 20 : -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-[75%] p-3 rounded-lg ${
              message.type === 'user' ? 'bg-blue-500 text-white' : 'bg-white text-blue-800 shadow-md'
            }`}>
              {message.content}
            </div>
          </motion.div>
        ))}
      </div>
      <div className="p-4 bg-white bg-opacity-50 backdrop-blur-sm">
        <div className="flex items-center space-x-2">
          <Input
            type="text"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAnswer()}
            placeholder={userName ? "Type your answer here..." : "Enter your name"}
            className="flex-grow bg-white bg-opacity-75"
          />
          <Button onClick={handleAnswer} disabled={isLoading} className="bg-blue-500 hover:bg-blue-600 text-white">
            <Send size={20} />
          </Button>
        </div>
        {!file && (
          <div className="mt-4 flex justify-center">
            <label htmlFor="file-upload" className="cursor-pointer">
              <div className="flex items-center space-x-2 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded transition duration-200">
                <Upload size={20} />
                <span>Upload PDF</span>
              </div>
              <input
                id="file-upload"
                type="file"
                onChange={handleFileUpload}
                accept=".pdf"
                className="hidden"
              />
            </label>
          </div>
        )}
        {questions.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-blue-800 mb-2">Progress</p>
            <Progress value={(currentQuestionIndex / questions.length) * 100} className="h-2 bg-blue-200" />
          </div>
        )}
      </div>
    </div>
  )
}

const HomePage = () => {
  const [isOnboardingOpen, setIsOnboardingOpen] = useState(true)
  const [isNewsFeedOpen, setIsNewsFeedOpen] = useState(true)
  const [isChatbotMaximized, setIsChatbotMaximized] = useState(false)
  const [newsItems, setNewsItems] = useState([])
  const [vulnerabilityReport, setVulnerabilityReport] = useState(null)
  const router = useRouter()

  useEffect(() => {
    const fetchedNews = [
      { id: 1, title: "Major data breach at tech giant", date: "2023-06-01" },
      { id: 2, title: "New ransomware strain targets healthcare", date: "2023-06-02" },
      { id: 3, title: "Critical vulnerability found in popular software", date: "2023-06-03" },
    ]
    setNewsItems(fetchedNews)
  }, [])

  const handleAssessmentComplete = async (results) => {
    router.push({
      pathname: '/assessment-results',
      query: { results: JSON.stringify(results) },
    })
  }

  const handleVulnerabilityReportGenerated = (report) => {
    setVulnerabilityReport(report)
  }

  const downloadReport = () => {
    const element = document.createElement("a")
    const file = new Blob([JSON.stringify(vulnerabilityReport, null, 2)], {type: 'text/plain'})
    element.href = URL.createObjectURL(file)
    element.download = "vulnerability_report.txt"
    document.body.appendChild(element)
    element.click()
  }


  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-white">
      <nav className="bg-white bg-opacity-70 backdrop-blur-lg shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/" className="flex items-center space-x-2">
                <Shield className="w-8 h-8 text-blue-600" />
                <span className="font-extrabold text-blue-700 text-xl">Cyber.AI</span>
              </Link>
            </div>
            <div className="flex items-center space-x-4">
              <Bell className="w-6 h-6 text-blue-600 cursor-pointer" />
              <UserButton />
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto mt-8 p-6 flex space-x-6">
        <motion.div
          className="w-1/3 space-y-6"
          initial={{ width: "33.333%" }}
          animate={{ width: isOnboardingOpen || isNewsFeedOpen ? "33.333%" : "0%" }}
          transition={{ duration: 0.3 }}
        >
          <AnimatePresence>
            {isOnboardingOpen && (
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-70 backdrop-blur-lg rounded-lg shadow-lg p-6"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold text-blue-800">Assessment Flow</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsOnboardingOpen(!isOnboardingOpen)}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    <ChevronLeft size={20} />
                  </Button>
                </div>
                <ol className="list-decimal list-inside space-y-4 text-blue-800">
                  <li className="flex items-center">
                    <span className="mr-2">📁</span>
                    Upload infrastructure documents
                  </li>
                  <li className="flex items-center">
                    <span className="mr-2">💬</span>
                    Answer questions about your infrastructure
                  </li>
                  <li className="flex items-center">
                    <span className="mr-2">📊</span>
                    Receive risk score and vulnerabilities report
                  </li>
                </ol>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {isNewsFeedOpen && (
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-70 backdrop-blur-lg rounded-lg shadow-lg p-6"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold text-blue-800">Incident Feed</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsNewsFeedOpen(!isNewsFeedOpen)}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    <ChevronLeft size={20} />
                  </Button>
                </div>
                <ul className="space-y-4">
                  {newsItems.map((item) => (
                    <li key={item.id} className="border-b border-blue-200 pb-4">
                      <p className="font-semibold text-blue-800">{item.title}</p>
                      <p className="text-sm text-blue-600">{item.date}</p>
                    </li>
                  ))}
                </ul>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.div
          className="flex-grow"
          animate={{ width: isChatbotMaximized ? "100%" : "66.666%" }}
          transition={{ duration: 0.3 }}
        >
          <div className="bg-white bg-opacity-70 backdrop-blur-lg rounded-lg shadow-lg h-[calc(100vh-8rem)] flex flex-col">
            <div  className="p-4 border-b border-blue-200 flex justify-between items-center">
              <h2 className="text-xl font-semibold text-blue-800">Cyber Bot</h2>
              <div className="flex space-x-2">
                {!isOnboardingOpen && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsOnboardingOpen(true)}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    <ChevronRight size={20} />
                  </Button>
                )}
                {!isNewsFeedOpen && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsNewsFeedOpen(true)}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    <ChevronRight size={20} />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsChatbotMaximized(!isChatbotMaximized)}
                  className="text-blue-600 hover:text-blue-800"
                >
                  {isChatbotMaximized ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
                </Button>
              </div>
            </div>
            <div className="flex-grow overflow-hidden">
              <BizBot onComplete={handleAssessmentComplete} onVulnerabilityReportGenerated={handleVulnerabilityReportGenerated} />
            </div>
          </div>
        </motion.div>
        {vulnerabilityReport && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white bg-opacity-70 backdrop-blur-lg rounded-lg shadow-lg p-6"
          >
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-blue-800">Vulnerability Report</h2>
              <Button onClick={downloadReport} className="flex items-center space-x-2 bg-blue-500 hover:bg-blue-600 text-white">
                <Download size={20} />
                <span>Download Report</span>
              </Button>
            </div>
            <pre className="bg-gray-100 p-4 rounded overflow-x-auto">
              {JSON.stringify(vulnerabilityReport, null, 2)}
            </pre>
          </motion.div>
      
        )}
        
      </div>
      <VulnerabilityReport />
    </div>
  )
}

export default HomePage